from pathlib import Path
import pandas as pd
import numpy as np

OUTPUT_DIR = Path('/Users/bitsaem/Desktop/stress_project/py_result')

# 1. 이미 저장된 결과 파일들 불러오기
sub_v9 = pd.read_csv(OUTPUT_DIR / 'submit_v9_balanced.csv')
sub_v11 = pd.read_csv(OUTPUT_DIR / 'submit_v11_golden.csv')
sub_v12 = pd.read_csv(OUTPUT_DIR / 'submit_v12_developed.csv')

# 2. 성능이 가장 좋았던 V11과 V12에 더 높은 가중치를 주어 블렌딩 (합이 1.0이 되도록 설정)
w_v9 = 0.2
w_v11 = 0.45
w_v12 = 0.35

blended_preds = (
    w_v9 * sub_v9['stress_score'] +
    w_v11 * sub_v11['stress_score'] +
    w_v12 * sub_v12['stress_score']
)

# 3. 최종 클리핑 및 저장
blended_preds = np.clip(np.round(blended_preds, 2), 0, 1)

final_sub = sub_v11.copy()
final_sub['stress_score'] = blended_preds

blend_path = OUTPUT_DIR / 'submit_blend_final.csv'
final_sub.to_csv(blend_path, index=False, encoding='utf-8')
print(f'🎯 블렌딩 파일 저장 완료: {blend_path}')