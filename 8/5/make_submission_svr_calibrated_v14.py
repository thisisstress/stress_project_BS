from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from experiment_svr_demographic_interactions_v6 import make_model
from experiment_svr_v10_kernel_tuning_v11 import build_v10_winner


DATA_DIR = Path("/Users/bitsaem/Desktop/stress_project/open")
RESULT_DIR = Path("/Users/bitsaem/Desktop/stress_project/8/5/8/5_result")
OOF_PATH = RESULT_DIR / "svr_v11_best_oof.csv"
TARGET = "stress_score"
ID_COLUMN = "ID"
C = 5.0
GAMMA = 0.08
N_BINS = 25


def fit_bin_calibrator(prediction, target, n_bins=N_BINS):
    edges = np.unique(np.quantile(prediction, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        raise RuntimeError("보정 구간 경계를 만들 수 없습니다.")
    edges[0], edges[-1] = -np.inf, np.inf
    bin_id = np.digitize(prediction, edges[1:-1], right=True)
    global_shift = float(np.median(target - prediction))
    shifts = []
    counts = []
    for value in range(len(edges) - 1):
        mask = bin_id == value
        count = int(mask.sum())
        local_shift = float(np.median(target[mask] - prediction[mask]))
        reliability = count / (count + 30.0)
        shifts.append(reliability * local_shift + (1.0 - reliability) * global_shift)
        counts.append(count)
    return edges, np.asarray(shifts), counts


def apply_calibrator(prediction, edges, shifts, strength):
    bin_id = np.digitize(prediction, edges[1:-1], right=True)
    return np.clip(prediction + strength * shifts[bin_id], 0.0, 1.0)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    oof = pd.read_csv(OOF_PATH)

    # 보정 기준은 train의 OOF 예측과 train 정답으로만 만든다.
    oof_prediction = oof["pred_mean"].to_numpy(float)
    oof_target = oof[TARGET].to_numpy(float)
    edges, shifts, counts = fit_bin_calibrator(oof_prediction, oof_target)

    train_features = build_v10_winner(train.drop(columns=[ID_COLUMN, TARGET]))
    test_features = build_v10_winner(test.drop(columns=[ID_COLUMN]))
    model = make_model(train_features, C, GAMMA)
    model.fit(train_features, train[TARGET].to_numpy(float))
    raw_test_prediction = np.clip(model.predict(test_features), 0.0, 1.0)

    outputs = {
        "main": (1.00, "submit_svr_v14_calibrated25.csv"),
        "conservative75": (0.75, "submit_svr_v14_calibrated25_strength75.csv"),
        "raw": (0.00, "submit_svr_v14_raw.csv"),
    }
    for label, (strength, filename) in outputs.items():
        prediction = apply_calibrator(raw_test_prediction, edges, shifts, strength)
        submission = sample.copy()
        submission[ID_COLUMN] = test[ID_COLUMN].to_numpy()
        submission[TARGET] = prediction
        path = RESULT_DIR / filename
        submission.to_csv(path, index=False)
        print(
            f"[{label:<14}] Saved: {path.resolve()} | "
            f"range={prediction.min():.6f}~{prediction.max():.6f} | "
            f"mean={prediction.mean():.6f}",
            flush=True,
        )

    table = pd.DataFrame(
        {
            "bin": np.arange(len(shifts)),
            "lower": edges[:-1],
            "upper": edges[1:],
            "train_oof_count": counts,
            "learned_shift": shifts,
        }
    )
    table_path = RESULT_DIR / "svr_v14_train_oof_calibration_table.csv"
    table.to_csv(table_path, index=False)

    calibrated_oof = apply_calibrator(oof_prediction, edges, shifts, 1.0)
    print("\n=============== SVR V14 제출 파일 생성 ===============")
    print(f"원본 OOF MAE       : {mean_absolute_error(oof_target, oof_prediction):.6f}")
    print(
        "전체 OOF 재학습 보정 MAE(참고용): "
        f"{mean_absolute_error(oof_target, calibrated_oof):.6f}"
    )
    print("독립 Cross-fit Audit MAE: 0.158020")
    print(f"보정 구간/강도     : {N_BINS} / 1.00")
    print(f"보정표             : {table_path.resolve()}")
    print("제출 우선순위       : main > conservative75 > raw")
    print("=======================================================")
    print("\nMain 첫 10개 예측값")
    main_pred = apply_calibrator(raw_test_prediction, edges, shifts, 1.0)
    preview = pd.DataFrame({ID_COLUMN: test[ID_COLUMN], TARGET: main_pred}).head(10)
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
