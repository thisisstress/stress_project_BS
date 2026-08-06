from __future__ import annotations

import argparse
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
from lightgbm import LGBMRegressor

TARGET = "stress_score"
ID_COLUMN = "ID"
RANDOM_SEED = 42
N_ESTIMATORS = 1200
TREE_QUANTILE = 0.54
PAIR_QUANTILE = 0.48
PAIR_WEIGHT = 0.28
LGBM_WEIGHT = 0.20  # 새로 추가된 LightGBM 모델의 반영 비율 (기존 모델들과 합산)

FEATURE_COPY_COUNTS = {
    "mean_working": 1,
    "bmi": 4,
    "cholesterol": 2,
    "height": 2,
    "glucose": 3,
    "weight": 2,
    "cholesterol_glucose_ratio": 2,
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

def validate_inputs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    submission: pd.DataFrame,
) -> None:
    if set(train.columns) - {TARGET} != set(test.columns):
        raise ValueError("Train과 Test의 입력 컬럼이 다릅니다.")
    if list(submission.columns) != [ID_COLUMN, TARGET]:
        raise ValueError("제출 양식 컬럼이 올바르지 않습니다.")
    if not submission[ID_COLUMN].equals(test[ID_COLUMN]):
        raise ValueError("제출 양식과 Test의 ID 순서가 다릅니다.")
    if train[TARGET].isna().any():
        raise ValueError("Train 타깃에 결측값이 있습니다.")

# 데이터 로드 및 준비
DATA_DIR = Path('/Users/bitsaem/Desktop/stress_project/open')
OUTPUT_DIR = Path('/Users/bitsaem/Desktop/stress_project')
train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')
submission = pd.read_csv(DATA_DIR / 'sample_submission.csv')
validate_inputs(train, test, submission)

X = add_features(train.drop(columns=[TARGET, ID_COLUMN])).reset_index(drop=True)
X_test = add_features(test.drop(columns=[ID_COLUMN])).reset_index(drop=True)
y = train[TARGET].reset_index(drop=True)
PAIRS = list(combinations(PAIR_COLUMNS, 2))

# 폴드별 예측 함수 정의
def run_fold_prediction(train_feat, val_feat, train_target, test_feat):
    # 1. Tree Quantile 예측
    aa = add_feature_copies(train_feat)
    val_aa = add_feature_copies(val_feat)
    test_aa = add_feature_copies(test_feat)
    
    pre = make_preprocessor(aa)
    am = pre.fit_transform(aa)
    vm = pre.transform(val_aa)
    tm = pre.transform(test_aa)
    
    et_model = ExtraTreesRegressor(
        n_estimators=1200, min_samples_leaf=1, max_features=1, random_state=RANDOM_SEED, n_jobs=-1
    )
    et_model.fit(am, train_target)
    
    val_tree_matrix = np.column_stack([tree.predict(vm) for tree in et_model.estimators_])
    val_tree_pred = np.quantile(val_tree_matrix, TREE_QUANTILE, axis=1)
    
    test_tree_matrix = np.column_stack([tree.predict(tm) for tree in et_model.estimators_])
    test_tree_pred = np.quantile(test_tree_matrix, TREE_QUANTILE, axis=1)
    
    # 2. LightGBM 예측 (MAE 손실함수 사용)
    lgb_model = LGBMRegressor(
        n_estimators=800, learning_rate=0.03, objective='regression_l1', random_state=RANDOM_SEED, n_jobs=-1, verbose=-1
    )
    lgb_model.fit(am, train_target)
    val_lgb_pred = lgb_model.predict(vm)
    test_lgb_pred = lgb_model.predict(tm)

    # 3. Pair Neighbor 예측
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
    
    # 4. 결합 (Ensemble)
    w_tree = 1.0 - PAIR_WEIGHT - LGBM_WEIGHT
    val_pred = w_tree * val_tree_pred + LGBM_WEIGHT * val_lgb_pred + PAIR_WEIGHT * val_pair_pred
    test_pred = w_tree * test_tree_pred + LGBM_WEIGHT * test_lgb_pred + PAIR_WEIGHT * test_pair_pred
    
    return np.clip(np.round(val_pred, 2), 0, 1), np.clip(np.round(test_pred, 2), 0, 1)

# 5-Fold Cross Validation 및 Test 앙상블 수행
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
oof = np.zeros(len(y))
test_preds = np.zeros(len(X_test))
rows = []

for fold, (ti, vi) in enumerate(cv.split(X), 1):
    X_train_fold, X_val_fold = X.iloc[ti], X.iloc[vi]
    y_train_fold, y_val_fold = y.iloc[ti], y.iloc[vi]
    
    val_pred, test_pred = run_fold_prediction(X_train_fold, X_val_fold, y_train_fold, X_test)
    
    oof[vi] = val_pred
    test_preds += test_pred / 5.0  # 5개 폴드의 테스트 예측값 평균
    
    score = mean_absolute_error(y_val_fold, val_pred)
    rows.append({'Fold': fold, 'MAE': score})
    print(f'Fold {fold} MAE: {score:.6f}')

print(f'OOF CV MAE: {mean_absolute_error(y, oof):.6f}')
print(pd.DataFrame(rows))

# 최종 파일 저장
output = submission.copy()
output[TARGET] = np.clip(np.round(test_preds, 2), 0, 1)
path = OUTPUT_DIR / 'submit_v35_lgb_ensemble.csv'
output.to_csv(path, index=False, encoding='utf-8')
print('Saved:', path)
print('Prediction range:', output[TARGET].min(), output[TARGET].max())
print("\n[상위 10개 예측 결과]")
print(output.head(10).to_string(index=False))
'''-----------
로컬 OOF MAE 점수 확인 및 오차 분석
-----------'''
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np

# 1. 5-Fold OOF(Out-Of-Fold) 전체 MAE 점수 계산
final_oof_mae = mean_absolute_error(y, oof)
print("=" * 50)
print(f"🎯 최종 로컬 OOF MAE 점수: {final_oof_mae:.6f}")
print("=" * 50)

# 2. 폴드별 성능 요약 데이터프레임 확인
cv_result_df = pd.DataFrame(rows)
print(cv_result_df)

# 3. 타깃(stress_score) 실제값 vs OOF 예측값 분포 비교 (오차 분석용)
pred_vs_actual = pd.DataFrame({
    'Actual': y,
    'OOF_Pred': oof,
    'Error': np.abs(y - oof)
})
print("\n[오차가 큰 상위 5개 샘플 분석 (Hard Examples)]")
print(pred_vs_actual.sort_values(by='Error', ascending=False).head(5))

print(f"\n예측값 범위 - 최소값: {oof.min():.4f}, 최대값: {oof.max():.4f}")
print(f"실제값 범위 - 최소값: {y.min():.4f}, 최대값: {y.max():.4f}")