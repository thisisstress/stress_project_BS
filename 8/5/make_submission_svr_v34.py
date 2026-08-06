from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from experiment_svr_binning_blend_v33 import (
    BIN_GROUPS,
    DATA_DIR,
    ID_COLUMN,
    RESULT_DIR,
    SEEDS,
    TARGET,
    build_v10_winner,
)
from experiment_svr_binning_strategy_v32 import make_model
from experiment_svr_v33_calibration_audit_v34 import bin_calibrate


GAMMA = 0.025
BIN_WEIGHT = 1.3
KMEANS_WEIGHT = 0.30
CALIBRATION_BINS = 12
CALIBRATION_SHRINK = 40.0


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    x_train = build_v10_winner(train.drop(columns=[ID_COLUMN, TARGET]))
    x_test = build_v10_winner(test.drop(columns=[ID_COLUMN]))
    x_test = x_test.reindex(columns=x_train.columns)
    target = train[TARGET].to_numpy(float)
    columns = tuple(sorted(
        (BIN_GROUPS["anthropometric"] | BIN_GROUPS["metabolic"])
        - {"age_bmi", "age_glucose"}
    ))

    test_predictions = []
    model_count = 0
    print("V34 Test fold-bagging: 3 seeds × 5 folds × 2 binning models")
    for seed in SEEDS:
        splitter = KFold(5, shuffle=True, random_state=seed)
        for fold, (train_idx, _) in enumerate(splitter.split(x_train), 1):
            quantile = make_model(
                x_train, GAMMA, columns, 4, BIN_WEIGHT, "quantile"
            )
            kmeans = make_model(
                x_train, GAMMA, columns, 5, BIN_WEIGHT, "kmeans"
            )
            quantile.fit(x_train.iloc[train_idx], target[train_idx])
            kmeans.fit(x_train.iloc[train_idx], target[train_idx])
            quantile_test = np.clip(quantile.predict(x_test), 0.0, 1.0)
            kmeans_test = np.clip(kmeans.predict(x_test), 0.0, 1.0)
            test_predictions.append(
                (1.0 - KMEANS_WEIGHT) * quantile_test
                + KMEANS_WEIGHT * kmeans_test
            )
            model_count += 2
            print(f"Seed {seed} | Fold {fold}/5 complete", flush=True)

    raw_prediction = np.clip(np.mean(test_predictions, axis=0), 0.0, 1.0)

    oof_path = RESULT_DIR / "svr_v34_v33_ensemble_oof.csv"
    oof = pd.read_csv(oof_path)
    oof_prediction = oof["pred_v33_ensemble"].to_numpy(float)
    oof_target = oof[TARGET].to_numpy(float)

    calibrated = np.clip(
        bin_calibrate(
            oof_prediction,
            oof_target,
            raw_prediction,
            CALIBRATION_BINS,
            1.00,
            CALIBRATION_SHRINK,
        ),
        0.0,
        1.0,
    )
    conservative = np.clip(
        bin_calibrate(
            oof_prediction,
            oof_target,
            raw_prediction,
            CALIBRATION_BINS,
            0.75,
            CALIBRATION_SHRINK,
        ),
        0.0,
        1.0,
    )

    outputs = {
        "main": (calibrated, RESULT_DIR / "submit_svr_v34_calibrated12.csv"),
        "conservative75": (
            conservative,
            RESULT_DIR / "submit_svr_v34_calibrated12_strength75.csv",
        ),
        "raw": (raw_prediction, RESULT_DIR / "submit_svr_v34_raw.csv"),
    }
    for name, (prediction, path) in outputs.items():
        submission = sample.copy()
        submission[ID_COLUMN] = test[ID_COLUMN].to_numpy()
        submission[TARGET] = prediction
        submission.to_csv(path, index=False)
        print(
            f"[{name:<14}] Saved: {path.resolve()} | "
            f"range={prediction.min():.6f}~{prediction.max():.6f} | "
            f"mean={prediction.mean():.6f}"
        )

    print("\n================ SVR V34 제출 파일 ================")
    print(f"학습 모델 수      : {model_count}")
    print("모델 혼합         : Quantile 70% + KMeans 30%")
    print("독립 Audit MAE    : 0.150811")
    print("독립 Audit 개선   : 7/7")
    print("제출 우선순위     : main > conservative75 > raw")
    print("====================================================")
    print("\nMain 첫 10개 예측값")
    preview = sample.copy()
    preview[ID_COLUMN] = test[ID_COLUMN].to_numpy()
    preview[TARGET] = calibrated
    print(preview.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
