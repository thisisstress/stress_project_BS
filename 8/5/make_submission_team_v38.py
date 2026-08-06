from pathlib import Path

import numpy as np
import pandas as pd

from experiment_team_agephys_selection_v38 import make_features
from experiment_team_feature_transfer_v35 import (
    DATA_DIR, ID_COLUMN, RESULT_DIR, TARGET, load_team_functions,
)


def model_prediction(features_train, features_test, y, tree_predict, pair_predict, pair_weight):
    tree = tree_predict(features_train, features_test, y)
    pair = pair_predict(features_train, features_test, y)
    # 기존 팀 모델과 동일하게 모델 내부 예측은 먼저 둘째 자리로 반올림한다.
    return np.clip(np.round((1 - pair_weight) * tree + pair_weight * pair, 2), 0, 1)


def save_submission(sample, predictions, filename):
    submission = sample.copy()
    submission[TARGET] = np.clip(predictions, 0, 1)
    path = RESULT_DIR / filename
    submission.to_csv(path, index=False)
    print(
        f"Saved: {path.resolve()} | range={submission[TARGET].min():.6f}~"
        f"{submission[TARGET].max():.6f} | mean={submission[TARGET].mean():.6f}"
    )
    return submission


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    add_features, tree_predict, pair_predict, pair_weight = load_team_functions()
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    raw_train = train.drop(columns=[ID_COLUMN, TARGET])
    raw_test = test.drop(columns=[ID_COLUMN])
    y = train[TARGET].reset_index(drop=True)

    age_columns = {"age_bmi", "age_map", "age_glucose", "age_bone_density"}
    glucose_bone_columns = {"age_glucose", "age_bone_density"}

    age_train = make_features(raw_train, add_features, age_columns)
    age_test = make_features(raw_test, add_features, age_columns)
    gb_train = make_features(raw_train, add_features, glucose_bone_columns)
    gb_test = make_features(raw_test, add_features, glucose_bone_columns)

    print("age_full4 모델을 전체 Train으로 학습합니다.", flush=True)
    age_pred = model_prediction(
        age_train, age_test, y, tree_predict, pair_predict, pair_weight
    )
    print("glucose_bone 모델을 전체 Train으로 학습합니다.", flush=True)
    gb_pred = model_prediction(
        gb_train, gb_test, y, tree_predict, pair_predict, pair_weight
    )

    # V38 평균 MAE 최저: glucose_bone 50% + age_full4 50%, 최종 round2.
    main_pred = np.round(0.50 * gb_pred + 0.50 * age_pred, 2)
    # V38 최악 시드 개선 최대: glucose_bone 25% + age_full4 75%, 반올림 없음.
    stable_pred = 0.25 * gb_pred + 0.75 * age_pred
    age_only_pred = age_pred

    main = save_submission(sample, main_pred, "submit_team_v38_main.csv")
    save_submission(sample, stable_pred, "submit_team_v38_stable25.csv")
    save_submission(sample, age_only_pred, "submit_team_v38_age_full4.csv")

    assert list(main.columns) == list(sample.columns)
    assert main[ID_COLUMN].equals(sample[ID_COLUMN])
    assert main[TARGET].notna().all()
    assert main[TARGET].between(0, 1).all()

    print("\n================ Team V38 제출 파일 ================")
    print("제출 1순위 : submit_team_v38_main.csv")
    print("구성       : glucose_bone 50% + age_full4 50% + Round2")
    print("OOF MAE    : 0.145808")
    print("승리 시드  : 3/3")
    print("보수 후보  : submit_team_v38_stable25.csv (OOF 0.145901)")
    print("=====================================================")
    print("\nMain 첫 10개 예측값")
    print(main.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
