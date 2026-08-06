from __future__ import annotations

import os
from itertools import combinations
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

optuna.logging.set_verbosity(optuna.logging.WARNING)

TARGET = "stress_score"
ID_COLUMN = "ID"
RANDOM_SEED = 2026

TREE_QUANTILE = 0.5215
PAIR_QUANTILE = 0.4915
PAIR_WEIGHT = 0.365

FEATURE_COPY_COUNTS = {
    "mean_working": 2,
    "bmi": 4,
    "cholesterol": 3,
    "height": 2,
    "glucose": 3,
    "weight": 2,
    "cholesterol_glucose_ratio": 3,
    "bone_density": 4,
}

PAIR_COLUMNS = [
    "mean_working",
    "bmi",
    "cholesterol",
    "height",
    "glucose",
    "weight",
    "cholesterol_glucose_ratio",
    "bone_density",
]

def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    height_m = result["height"] / 100.0
    sbp = result["systolic_blood_pressure"]
    dbp = result["diastolic_blood_pressure"]
    result["bmi"] = result["weight"] / height_m.pow(2)
    result["pulse_pressure"] = sbp - dbp
    result["mean_arterial_pressure"] = (sbp + 2.0 * dbp) / 3.0
    result["blood_pressure_ratio"] = sbp / (dbp + 1e-6)
    result["cholesterol_glucose_ratio"] = result["cholesterol"] / (
        result["glucose"] + 1e-6
    )
    result["missing_count"] = result.isna().sum(axis=1)
    return result

def add_feature_copies(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for feature, additional_copies in FEATURE_COPY_COUNTS.items():
        for copy_index in range(1, additional_copies + 1):
            result[f"{feature}__copy{copy_index}"] = result[feature]
    return result

def make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    categorical = frame.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    numerical = frame.select_dtypes(include=["number"]).columns.tolist()
    return ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant", fill_value="missing"
                            ),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
            (
                "numerical",
                SimpleImputer(strategy="median", add_indicator=True),
                numerical,
            ),
        ]
    )

def empirical_rank_transform(
    train_values: np.ndarray,
    test_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(train_values)
    train_values = np.nan_to_num(train_values, nan=median)
    test_values = np.nan_to_num(test_values, nan=median)
    sorted_train = np.sort(train_values)
    train_rank = np.searchsorted(
        sorted_train, train_values, side="right"
    ) / len(train_values)
    test_rank = np.searchsorted(
        sorted_train, test_values, side="right"
    ) / len(train_values)
    return train_rank.astype(np.float32), test_rank.astype(np.float32)

DATA_DIR = Path('/Users/bitsaem/Desktop/stress_project/open')
OUTPUT_DIR = Path('/Users/bitsaem/Desktop/stress_project/py_result')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')
submission = pd.read_csv(DATA_DIR / 'sample_submission.csv')

# 노이즈 필터링 적용
lower_bound = train[TARGET].quantile(0.01)
upper_bound = train[TARGET].quantile(0.99)
train = train[(train[TARGET] >= lower_bound) & (train[TARGET] <= upper_bound)].reset_index(drop=True)

X = add_features(train.drop(columns=[TARGET, ID_COLUMN])).reset_index(drop=True)
X_test = add_features(test.drop(columns=[ID_COLUMN])).reset_index(drop=True)
y = train[TARGET].reset_index(drop=True)
PAIRS = list(combinations(PAIR_COLUMNS, 2))

# Optuna로 최적의 n_estimators 및 트리 파라미터 탐색
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 1000, 2000, step=200)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 3)
    max_features_val = trial.suggest_float('max_features_val', 0.8, 1.0)
    
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    
    for ti, vi in cv.split(X):
        X_tr_f, X_va_f = X.iloc[ti], X.iloc[vi]
        y_tr_f, y_va_f = y.iloc[ti], y.iloc[vi]
        
        aa = add_feature_copies(X_tr_f)
        val_aa = add_feature_copies(X_va_f)
        
        pre = make_preprocessor(aa)
        am = pre.fit_transform(aa)
        vm = pre.transform(val_aa)
        
        et_model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features_val,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        et_model.fit(am, y_tr_f)
        
        val_tree_matrix = np.column_stack([tree.predict(vm) for tree in et_model.estimators_])
        val_tree_pred = np.quantile(val_tree_matrix, TREE_QUANTILE, axis=1)
        scores.append(mean_absolute_error(y_va_f, val_tree_pred))
        
    return np.mean(scores)

print("🔥 [V18] Optuna 최적 하이퍼파라미터 탐색 시작...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10) # 시간이 부족하므로 10회 빠르게 탐색

print("=" * 50)
print(f"🎯 최적 탐색 파라미터: {study.best_params}")
print("=" * 50)

best_n_est = study.best_params['n_estimators']
best_min_leaf = study.best_params['min_samples_leaf']
best_max_feat = study.best_params['max_features_val']

def run_fold_prediction_optuna(train_feat, val_feat, train_target, test_feat):
    aa = add_feature_copies(train_feat)
    val_aa = add_feature_copies(val_feat)
    test_aa = add_feature_copies(test_feat)
    
    pre = make_preprocessor(aa)
    am = pre.fit_transform(aa)
    vm = pre.transform(val_aa)
    tm = pre.transform(test_aa)
    
    et_model = ExtraTreesRegressor(
        n_estimators=best_n_est, 
        min_samples_leaf=best_min_leaf, 
        max_features=best_max_feat, 
        random_state=RANDOM_SEED, 
        n_jobs=-1
    )
    et_model.fit(am, train_target)
    
    val_tree_matrix = np.column_stack([tree.predict(vm) for tree in et_model.estimators_])
    val_tree_pred = np.quantile(val_tree_matrix, TREE_QUANTILE, axis=1)
    
    test_tree_matrix = np.column_stack([tree.predict(tm) for tree in et_model.estimators_])
    test_tree_pred = np.quantile(test_tree_matrix, TREE_QUANTILE, axis=1)
    
    tr_rank, val_rank, test_rank = {}, {}, {}
    for c in PAIR_COLUMNS:
        tr_rank[c], val_rank[c] = empirical_rank_transform(train_feat[c].to_numpy(float), val_feat[c].to_numpy(float))
        _, test_rank[c] = empirical_rank_transform(train_feat[c].to_numpy(float), test_feat[c].to_numpy(float))
        
    tv = train_target.to_numpy(float)
    val_values, test_values = [], []
    for c1, c2 in PAIRS:
        d_val = np.abs(val_rank[c1][:, None] - tr_rank[c1][None, :]) + np.abs(val_rank[c2][:, None] - tr_rank[c2][None, :])
        val_values.append(tv[np.argmin(d_val, axis=1)])
        
        d_test = np.abs(test_rank[c1][:, None] - tr_rank[c1][None, :]) + np.abs(test_rank[c2][:, None] - tr_rank[c2][None, :])
        test_values.append(tv[np.argmin(d_test, axis=1)])
        
    val_pair_pred = np.quantile(np.column_stack(val_values), PAIR_QUANTILE, axis=1)
    test_pair_pred = np.quantile(np.column_stack(test_values), PAIR_QUANTILE, axis=1)
    
    val_pred = (1.0 - PAIR_WEIGHT) * val_tree_pred + PAIR_WEIGHT * val_pair_pred
    test_pred = (1.0 - PAIR_WEIGHT) * test_tree_pred + PAIR_WEIGHT * test_pair_pred
    
    return np.clip(np.round(val_pred, 2), 0, 1), np.clip(np.round(test_pred, 2), 0, 1)

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
oof = np.zeros(len(y))
test_preds = np.zeros(len(X_test))

print("🔥 [V18 최종 모델] 교차 검증 학습 시작...")
for fold, (ti, vi) in enumerate(cv.split(X), 1):
    X_train_fold, X_val_fold = X.iloc[ti], X.iloc[vi]
    y_train_fold, y_val_fold = y.iloc[ti], y.iloc[vi]
    
    val_pred, test_pred = run_fold_prediction_optuna(X_train_fold, X_val_fold, y_train_fold, X_test)
    
    oof[vi] = val_pred
    test_preds += test_pred / 5.0  
    
    score = mean_absolute_error(y_val_fold, val_pred)
    print(f'Fold {fold} MAE: {score:.6f}')

final_oof_mae = mean_absolute_error(y, oof)
print("=" * 50)
print(f"🎯 [V18 최종] 노이즈 제거 + Optuna OOF MAE 점수: {final_oof_mae:.6f}")
print("=" * 50)

output = submission.copy()
output[TARGET] = np.clip(np.round(test_preds, 2), 0, 1)
path = OUTPUT_DIR / 'submit_v18_optuna_noise.csv'
output.to_csv(path, index=False, encoding='utf-8')
print(f'\nSaved: {path}')