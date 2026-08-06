from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

OUTPUT_DIR = Path('/Users/bitsaem/Desktop/stress_project/py_result')
SUBMISSION_PATH = Path('/Users/bitsaem/Desktop/stress_project/open/sample_submission.csv')

# 지금까지 가장 성능이 좋았던 핵심 결과 파일들 불러오기
file_v17 = OUTPUT_DIR / 'submit_v17_noise_filtered.csv'
file_v19 = OUTPUT_DIR / 'submit_v19_boost.csv'

df_v17 = pd.read_csv(file_v17)
df_v19 = pd.read_csv(file_v19)
submission = pd.read_csv(SUBMISSION_PATH)

TARGET = "stress_score"

print("🔥 최종 파일 블렌딩(가중 평균 앙상블) 진행 중...")

# 가장 점수가 좋았던 V17과 V19 파일의 예측값을 최적의 비율(예: 0.5대 0.5 또는 0.4대 0.6)로 결합
# 두 모델 모두 노이즈 필터링이 강력하게 적용되어 있으므로 동등한 비중으로 섞습니다.
blended_preds = (
    0.5 * df_v17[TARGET] + 
    0.5 * df_v19[TARGET]
)

# 최종 제출 양식에 맞게 값 조정 및 저장
submission[TARGET] = np.clip(np.round(blended_preds, 2), 0, 1)
final_path = OUTPUT_DIR / 'submit_final_blending_013.csv'
submission.to_csv(final_path, index=False, encoding='utf-8')

print("=" * 50)
print(f"🎯 최종 블렌딩 완료! 제출 파일 생성됨: {final_path}")
print("=" * 50)