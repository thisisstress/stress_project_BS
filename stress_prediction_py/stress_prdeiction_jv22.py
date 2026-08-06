from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 경로 설정
DATA_DIR = Path('/Users/bitsaem/Desktop/stress_project/open')
OUTPUT_DIR = Path('/Users/bitsaem/Desktop/stress_project/py_result')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("🔥 [최종 결전] 0.1265 재현 모델 학습 시작...")

train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')
submission = pd.read_csv(DATA_DIR / 'sample_submission.csv')

TARGET = "stress_score"

# 1. 범주형 및 수치형 데이터 전처리 자동화 (피처가 9개로 쪼그라드는 현상 원천 차단)
df_all = pd.concat([train.drop(columns=[TARGET], errors='ignore'), test], axis=0).reset_index(drop=True)

# 범주형 컬럼 인코딩
cat_cols = df_all.select_dtypes(include=['object', 'category']).columns.tolist()
if 'id' in cat_cols: cat_cols.remove('id')

for col in cat_cols:
    df_all[col] = df_all[col].astype('category').cat.codes

# 다시 train / test 분리
train_df = df_all.iloc[:len(train)].copy()
train_df[TARGET] = train[TARGET].values
test_df = df_all.iloc[len(train):].copy()

features = [col for col in train_df.columns if col not in ['id', TARGET]]

X = train_df[features]
y = train_df[TARGET]
X_test = test_df[features]

print(f"✅ 정상 반영된 학습 피처 개수: {len(features)}개")

# 2. 강력한 K-Fold 기반 ExtraTrees 앙상블 학습
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

print("🌲 ExtraTrees 모델 학습 중...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    model = ExtraTreesRegressor(
        n_estimators=600,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42 + fold
    )
    
    model.fit(X_tr, y_tr)
    oof_preds[val_idx] = model.predict(X_va)
    test_preds += model.predict(X_test) / N_SPLITS

print(f"🎯 최종 OOF MAE 점수: {mean_absolute_error(y, oof_preds):.6f}")

# 3. 최종 제출 파일 저장
submission[TARGET] = np.clip(np.round(test_preds, 4), 0, 1)
final_path = OUTPUT_DIR / 'submit_absolute_win.csv'
submission.to_csv(final_path, index=False, encoding='utf-8')

print("=" * 50)
print(f"🚀 최종 역전 제출 파일 생성 완료: {final_path}")
print("=" * 50)