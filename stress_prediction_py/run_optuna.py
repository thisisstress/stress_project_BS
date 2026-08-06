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
import optuna

# 로그 출력을 깔끔하게 설정
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

def objective(trial):
    # Optuna가 탐색할 하이퍼파라미터 범위 설정
    n_estimators = trial.suggest_int('n_estimators', 800, 2000, step=200)
    max_depth = trial.suggest_int('max_depth', 20, 50, step=5)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
    max_features = trial.suggest_float('max_features', 0.5, 1.0)
    
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED) # 속도를 위해 3-Fold로 빠른 탐색
    scores = []
    
    for ti, vi in cv.split(X_m):
        X_tr, X_va = X_m[ti], X_m[vi]
        y_tr, y_va = y.iloc[ti], y.iloc[vi]
        
        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        scores.append(mean_absolute_error(y_va, preds))
        
    return np.mean(scores)

print("🔥 [Optuna] 1등을 잡기 위한 최적의 하이퍼파라미터 탐색 시작...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20) # 20번 시도 (시간에 따라 n_trials를 늘려보세요!)

print("=" * 50)
print(f"🎯 최적의 파라미터 조합 발견: {study.best_params}")
print(f"🎯 최적 탐색 MAE 점수: {study.best_value:.6f}")
print("=" * 50)

# 찾은 최적의 파라미터로 전체 데이터 학습 및 최종 제출 파일 생성
print("🔥 최적 파라미터로 최종 모델 학습 및 예측 중...")
best_p = study.best_params

cv_full = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
test_preds = np.zeros(len(X_test))
oof_preds = np.zeros(len(y))

for ti, vi in cv_full.split(X_m):
    X_tr, X_va = X_m[ti], X_m[vi]
    y_tr, y_va = y.iloc[ti], y.iloc[vi]
    
    model = ExtraTreesRegressor(**best_p, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_tr, y_tr)
    
    oof_preds[vi] = model.predict(X_va)
    test_preds += model.predict(X_test_m) / 5.0

print(f"🎯 최종 5-Fold OOF MAE: {mean_absolute_error(y, oof_preds):.6f}")

output = submission.copy()
output[TARGET] = np.clip(np.round(test_preds, 2), 0, 1)
path = OUTPUT_DIR / 'submit_optuna_best.csv'
output.to_csv(path, index=False, encoding='utf-8')
print(f'Saved: {path}')