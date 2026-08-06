from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from experiment_svr_binning_blend_v33 import (
    BIN_GROUPS,
    DATA_DIR,
    ID_COLUMN,
    RESULT_DIR,
    SEEDS,
    TARGET,
    build_v10_winner,
    paired_oof,
)


KMEANS_WEIGHT = 0.30
AUDIT_SEEDS = (11, 29, 91, 333, 909, 4096, 8128)


def bin_calibrate(train_pred, train_y, valid_pred, n_bins, strength, shrink):
    edges = np.unique(np.quantile(train_pred, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return valid_pred.copy()
    edges[0], edges[-1] = -np.inf, np.inf
    train_bin = np.digitize(train_pred, edges[1:-1], right=True)
    valid_bin = np.digitize(valid_pred, edges[1:-1], right=True)
    global_shift = float(np.median(train_y - train_pred))
    shifts = np.zeros(len(edges) - 1, dtype=float)
    for bin_id in range(len(shifts)):
        mask = train_bin == bin_id
        if not np.any(mask):
            shifts[bin_id] = strength * global_shift
            continue
        local_shift = float(np.median(train_y[mask] - train_pred[mask]))
        reliability = mask.sum() / (mask.sum() + shrink)
        shifts[bin_id] = strength * (
            reliability * local_shift + (1.0 - reliability) * global_shift
        )
    return valid_pred + shifts[valid_bin]


def crossfit(prediction, target, seed, n_bins, strength, shrink):
    calibrated = np.zeros(len(target), dtype=float)
    splitter = KFold(5, shuffle=True, random_state=seed)
    for train_idx, valid_idx in splitter.split(prediction):
        calibrated[valid_idx] = bin_calibrate(
            prediction[train_idx], target[train_idx], prediction[valid_idx],
            n_bins, strength, shrink,
        )
    return mean_absolute_error(target, np.clip(calibrated, 0.0, 1.0))


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    features = build_v10_winner(train.drop(columns=[ID_COLUMN, TARGET]))
    target = train[TARGET].to_numpy(float)
    columns = tuple(sorted(
        (BIN_GROUPS["anthropometric"] | BIN_GROUPS["metabolic"])
        - {"age_bmi", "age_glucose"}
    ))

    seed_blends = []
    for seed in SEEDS:
        quantile_oof, kmeans_oof = paired_oof(features, target, seed, columns)
        seed_blends.append(
            (1.0 - KMEANS_WEIGHT) * quantile_oof + KMEANS_WEIGHT * kmeans_oof
        )
    prediction = np.mean(seed_blends, axis=0)
    baseline = mean_absolute_error(target, prediction)

    oof_path = RESULT_DIR / "svr_v34_v33_ensemble_oof.csv"
    pd.DataFrame({
        ID_COLUMN: train[ID_COLUMN],
        TARGET: target,
        "pred_v33_ensemble": prediction,
    }).to_csv(oof_path, index=False)

    configs = [
        (n_bins, strength, shrink)
        for n_bins in (8, 10, 12, 15, 20, 25)
        for strength in (0.50, 0.75, 1.00)
        for shrink in (20.0, 40.0, 80.0)
    ]
    rows = []
    print(f"V33 3-seed ensemble OOF MAE: {baseline:.6f}")
    print(f"V34 calibration audit: {len(configs)} configs × 7 seeds")
    for index, (n_bins, strength, shrink) in enumerate(configs, 1):
        scores = [
            crossfit(prediction, target, seed, n_bins, strength, shrink)
            for seed in AUDIT_SEEDS
        ]
        improvements = [baseline - score for score in scores]
        rows.append({
            "n_bins": n_bins,
            "strength": strength,
            "shrink": shrink,
            "mean_MAE": float(np.mean(scores)),
            "worst_MAE": float(np.max(scores)),
            "mean_improvement": float(np.mean(improvements)),
            "worst_improvement": float(np.min(improvements)),
            "wins": int(sum(value > 0 for value in improvements)),
        })
        print(
            f"[{index:02d}/{len(configs):02d}] bins={n_bins:<2} "
            f"strength={strength:.2f} shrink={shrink:<4.0f} | "
            f"평균={np.mean(scores):.6f} | 최악 개선={np.min(improvements):+.6f} | "
            f"승리={sum(value > 0 for value in improvements)}/7",
            flush=True,
        )

    result = pd.DataFrame(rows).sort_values(
        ["wins", "worst_improvement", "mean_MAE"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    result_path = RESULT_DIR / "svr_v34_v33_calibration_audit.csv"
    result.to_csv(result_path, index=False)
    best = result.iloc[0]

    print("\n================ SVR V34 감사 결과 ================")
    print(f"기준 Ensemble OOF: {baseline:.6f}")
    print(f"구간/강도/수축   : {int(best.n_bins)} / {best.strength:.2f} / {best.shrink:.0f}")
    print(f"평균 Audit MAE   : {best.mean_MAE:.6f}")
    print(f"평균 개선        : {best.mean_improvement:+.6f}")
    print(f"최악 개선        : {best.worst_improvement:+.6f}")
    print(f"개선 분할        : {int(best.wins)}/7")
    print("====================================================")
    print(f"OOF    : {Path(oof_path).resolve()}")
    print(f"Results: {Path(result_path).resolve()}")


if __name__ == "__main__":
    main()
