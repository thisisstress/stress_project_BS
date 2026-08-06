from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

OUTPUT_DIR = Path('/Users/bitsaem/Desktop/stress_project/py_result')
SUBMISSION_PATH = Path('/Users/bitsaem/Desktop/stress_project/open/sample_submission.csv')
TARGET = "stress_score"

# 1. 파일 불러오기
# 주의: '가장 점수가 높았던 기존 최고 파일명'을 아래 best_file 변수에 정확히 적어주세요!
# 예시: 'submit_best_ever.csv' 또는 본인이 0.1265를 얻었던 파일명
best_file_path = '/Users/bitsaem/Desktop/stress_project/submit_v34_joint_q54_q48_w28.csv' 
v17_file_path = OUTPUT_DIR / 'submit_v17_noise_filtered.csv'

# 만약 최고 파일명이 기억 안 나시면 py_result 폴더 안의 파일명을 확인 후 수정해주세요!
try:
    df_best = pd.read_csv(best_file_path)
    print("🔥 기존 최고 점수 파일 로드 성공!")
except Exception as e:
    print(f"⚠️ 기존 최고 파일을 못 찾았습니다. 파일명을 확인해주세요: {e}")
    exit()

df_v17 = pd.read_csv(v17_file_path)
submission = pd.read_csv(SUBMISSION_PATH)

print("🔥 1등 탈환을 위한 황금 비율 블렌딩(가중 평균) 진행 중...")

# 2. 황금 가중치 적용 
# 0.1265를 만든 최고 파일의 뼈대(85%)를 강력하게 유지하면서, 
# 노이즈가 걸러진 V17의 정교한 예측값(15%)을 미세하게 양념처럼 섞어줍니다.
weight_best = 0.85
weight_v17 = 0.15

blended_preds = (
    weight_best * df_best[TARGET] + 
    weight_v17 * df_v17[TARGET]
)

# 3. 최종 값 정리 및 저장
submission[TARGET] = np.clip(np.round(blended_preds, 2), 0, 1)
final_path = OUTPUT_DIR / 'submit_ultimate_1st.csv'
submission.to_csv(final_path, index=False, encoding='utf-8')

print("=" * 50)
print(f"🎯 최종 역전 제출 파일 생성 완료!: {final_path}")
print("이 파일을 해커톤 페이지에 업로드하고 1등을 탈환해 보세요!")
print("=" * 50)