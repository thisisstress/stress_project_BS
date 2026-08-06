from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from experiment_svr_block_weighting_v17 import make_block_weighted_model
from experiment_svr_v10_kernel_tuning_v11 import build_v10_winner


PROJECT_DIR = Path("/Users/bitsaem/Desktop/stress_project")
DATA_DIR = PROJECT_DIR / "open"
RESULT_DIR = PROJECT_DIR / "8/5/8/5_result"
TARGET = "stress_score"
ID_COLUMN = "ID"
SEEDS = (42, 77, 2026)
C = 5.0
GAMMA = 0.035
CAT_WEIGHT = 5.0


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    x_train = build_v10_winner(train.drop(columns=[ID_COLUMN, TARGET]))
    x_test = build_v10_winner(test.drop(columns=[ID_COLUMN]))
    target = train[TARGET].to_numpy(float)

    seed_oofs = []
    test_predictions = []
    fold_rows = []
    print("V19 SVR 3개 시드 × 5-Fold bagging을 시작합니다.")
    for seed in SEEDS:
        oof = np.zeros(len(train), dtype=float)
        splitter = KFold(5, shuffle=True, random_state=seed)
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(x_train), 1):
            model = make_block_weighted_model(x_train, C, GAMMA, CAT_WEIGHT)
            model.fit(x_train.iloc[train_idx], target[train_idx])
            valid_pred = np.clip(model.predict(x_train.iloc[valid_idx]), 0.0, 1.0)
            test_pred = np.clip(model.predict(x_test), 0.0, 1.0)
            oof[valid_idx] = valid_pred
            test_predictions.append(test_pred)
            score = mean_absolute_error(target[valid_idx], valid_pred)
            fold_rows.append({"seed": seed, "fold": fold, "MAE": score})
            print(f"Seed {seed} | Fold {fold} MAE: {score:.6f}", flush=True)
        seed_oofs.append(oof)
        print(f"Seed {seed} OOF MAE: {mean_absolute_error(target, oof):.6f}\n")

    oof_matrix = np.column_stack(seed_oofs)
    oof_mean = oof_matrix.mean(axis=1)
    test_matrix = np.column_stack(test_predictions)
    test_mean = test_matrix.mean(axis=1)

    oof_output = pd.DataFrame(
        {
            ID_COLUMN: train[ID_COLUMN],
            TARGET: target,
            **{f"oof_seed_{seed}": oof_matrix[:, i] for i, seed in enumerate(SEEDS)},
            "oof_mean": oof_mean,
        }
    )
    oof_path = RESULT_DIR / "svr_v23_foldbag_oof.csv"
    fold_path = RESULT_DIR / "svr_v23_foldbag_scores.csv"
    oof_output.to_csv(oof_path, index=False)
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)

    raw = sample.copy()
    raw[ID_COLUMN] = test[ID_COLUMN].to_numpy()
    raw[TARGET] = test_mean
    raw_path = RESULT_DIR / "submit_svr_v23_foldbag_raw.csv"
    raw.to_csv(raw_path, index=False)

    rounded = raw.copy()
    rounded[TARGET] = np.round(test_mean, 2)
    rounded_path = RESULT_DIR / "submit_svr_v23_foldbag_round2.csv"
    rounded.to_csv(rounded_path, index=False)

    print("\n================ SVR V23 Fold Bagging ================")
    print(f"학습 모델 수       : {test_matrix.shape[1]}")
    print(f"OOF 평균 MAE       : {mean_absolute_error(target, oof_mean):.6f}")
    print(
        f"OOF Round2 MAE     : {mean_absolute_error(target, np.round(oof_mean, 2)):.6f}"
    )
    print(f"Test 예측 범위     : {test_mean.min():.6f}~{test_mean.max():.6f}")
    print(f"Test 예측 평균     : {test_mean.mean():.6f}")
    print(f"OOF                : {oof_path.resolve()}")
    print(f"Fold 점수          : {fold_path.resolve()}")
    print(f"Raw 제출           : {raw_path.resolve()}")
    print(f"Round2 제출        : {rounded_path.resolve()}")
    print("======================================================")
    print("\nRaw 첫 10개 예측값")
    print(raw.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
