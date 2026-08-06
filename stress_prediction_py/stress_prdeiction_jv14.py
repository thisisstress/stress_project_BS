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
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import lightgbm as lgb

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

print("🔥 [LightGBM + ExtraTrees 앙상블] 교차 검증 시작...")
for fold, (ti, vi) in enumerate(cv.split(X_m), 1):
    X_tr, X_va = X_m[ti], X_m[vi]
    y_tr, y_va = y.iloc[ti], y.iloc[vi]
    
    # 1. ExtraTrees 학습
    et = ExtraTreesRegressor(n_estimators=1000, random_state=RANDOM_SEED, n_jobs=-1)
    et.fit(X_tr, y_tr)
    et_va = et.predict(X_va)
    et_te = et.predict(X_test_m)
    
    # 2. LightGBM 학습
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.03, random_state=RANDOM_SEED, verbose=-1
    )
    model_lgb.fit(X_tr, y_tr)
    lgb_va = model_lgb.predict(X_va)
    lgb_te = model_lgb.predict(X_test_m)
    
    # 3. 두 모델 예측값 앙상블 (ET 60%, LGB 40%)
    val_pred = 0.6 * et_va + 0.4 * lgb_va
    test_pred = 0.6 * et_te + 0.4 * lgb_te
    
    oof[vi] = val_pred
    test_preds += test_pred / 5.0
    print(f"Fold {fold} MAE: {mean_absolute_error(y_va, val_pred):.6f}")

print("=" * 50)
print(f"🎯 [LGB+ET 앙상블] 최종 OOF MAE: {mean_absolute_error(y, oof):.6f}")
print("=" * 50)

output = submission.copy()
output[TARGET] = np.clip(np.round(test_preds, 2), 0, 1)
path = OUTPUT_DIR / 'submit_lgb_et_ensemble.csv'
output.to_csv(path, index=False, encoding='utf-8')
print(f'Saved: {path}')