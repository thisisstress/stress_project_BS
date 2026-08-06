from __future__ import annotations
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

TARGET = "stress_score"
ID_COLUMN = "ID"
RANDOM_SEED = 2026

def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    height_m = result["height"] / 100.0
    sbp = result["systolic_blood_pressure"]
    dbp = result["diastolic_blood_pressure"]
    result["bmi"] = result["weight"] / height_m.pow(2)
    result["pulse_pressure"] = sbp - dbp
    result["mean_arterial_pressure"] = (sbp + 2.0 * dbp) / 3.0
    result["blood_pressure_ratio"] = sbp / (dbp + 1e-6)
    result["cholesterol_glucose_ratio"] = result["cholesterol"] / (result["glucose"] + 1e-6)
    result["missing_count"] = result.isna().sum(axis=1)
    return result

def make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    categorical = frame.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numerical = frame.select_dtypes(include=["number"]).columns.tolist()
    return ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical),
        ("numerical", SimpleImputer(strategy="median", add_indicator=True), numerical),
    ])

DATA_DIR = Path('/Users/bitsaem/Desktop/stress_project/open')
OUTPUT_DIR = Path('/Users/bitsaem/Desktop/stress_project/py_result')

train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')
submission = pd.read_csv(DATA_DIR / 'sample_submission.csv')

X = add_features(train.drop(columns=[TARGET, ID_COLUMN])).reset_index(drop=True)
X_test = add_features(test.drop(columns=[ID_COLUMN])).reset_index(drop=True)
y = train[TARGET].reset_index(drop=True)

pre = make_preprocessor(X)
X_m = pre.fit_transform(X)
X_test_m = pre.transform(X_test)

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
oof = np.zeros(len(y))
test_preds = np.zeros(len(X_test))

print("🔥 [타깃 스케일링 + 후처리 강화] 교차 검증 시작...")
for fold, (ti, vi) in enumerate(cv.split(X_m), 1):
    X_tr, X_va = X_m[ti], X_m[vi]
    y_tr, y_va = y.iloc[ti], y.iloc[vi]
    
    # 타깃 스케일링 적용
    scaler = StandardScaler()
    y_tr_scaled = scaler.fit_transform(y_tr.to_numpy().reshape(-1, 1)).ravel()
    
    et = ExtraTreesRegressor(n_estimators=1200, random_state=RANDOM_SEED, n_jobs=-1)
    et.fit(X_tr, y_tr_scaled)
    
    # 예측 후 역스케일링(Inverse Transform)
    val_pred_scaled = et.predict(X_va)
    test_pred_scaled = et.predict(X_test_m)
    
    val_pred = scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).ravel()
    test_pred = scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).ravel()
    
    oof[vi] = val_pred
    test_preds += test_pred / 5.0
    print(f"Fold {fold} MAE: {mean_absolute_error(y_va, val_pred):.6f}")

print("=" * 50)
print(f"🎯 [스케일링 적용] 최종 OOF MAE: {mean_absolute_error(y, oof):.6f}")
print("=" * 50)

output = submission.copy()
# 엄격한 후처리 및 클리핑 적용
final_vals = np.clip(test_preds, train[TARGET].min(), train[TARGET].max())
output[TARGET] = np.clip(np.round(final_vals, 2), 0, 1)

path = OUTPUT_DIR / 'submit_scaled_postprocess.csv'
output.to_csv(path, index=False, encoding='utf-8')
print(f'Saved: {path}')