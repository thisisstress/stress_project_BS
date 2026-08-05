from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


RESULT_DIR = Path("/Users/bitsaem/Desktop/stress_project/8/5/8/5_result")
OOF_PATH = RESULT_DIR / "svr_v11_best_oof.csv"
AUDIT_SEEDS = (11, 29, 91, 333, 909, 4096, 8128)


def bin_calibrate(train_pred, train_y, valid_pred, n_bins, strength):
    edges = np.unique(np.quantile(train_pred, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return valid_pred
    edges[0], edges[-1] = -np.inf, np.inf
    train_bin = np.digitize(train_pred, edges[1:-1], right=True)
    valid_bin = np.digitize(valid_pred, edges[1:-1], right=True)
    global_shift = float(np.median(train_y - train_pred))
    shifts = {}
    for bin_id in range(len(edges) - 1):
        mask = train_bin == bin_id
        local = float(np.median(train_y[mask] - train_pred[mask]))
        # 작은 구간은 전체 편향 쪽으로 수축해 불안정한 보정을 줄인다.
        reliability = mask.sum() / (mask.sum() + 30.0)
        shift = reliability * local + (1.0 - reliability) * global_shift
        shifts[bin_id] = strength * shift
    return valid_pred + np.array([shifts[value] for value in valid_bin])


def crossfit(prediction, target, seed, n_bins, strength):
    calibrated = np.zeros(len(target), dtype=float)
    splitter = KFold(5, shuffle=True, random_state=seed)
    for train_idx, valid_idx in splitter.split(prediction):
        calibrated[valid_idx] = bin_calibrate(
            prediction[train_idx],
            target[train_idx],
            prediction[valid_idx],
            n_bins,
            strength,
        )
    calibrated = np.clip(calibrated, 0.0, 1.0)
    return mean_absolute_error(target, calibrated)


def main() -> None:
    oof = pd.read_csv(OOF_PATH)
    prediction = oof["pred_mean"].to_numpy(float)
    target = oof["stress_score"].to_numpy(float)
    baseline = mean_absolute_error(target, prediction)

    configs = [
        (n_bins, strength)
        for n_bins in (8, 10, 12, 15, 18, 20, 25, 30)
        for strength in (0.50, 0.75, 1.00)
    ]
    rows = []
    print(f"Baseline OOF MAE: {baseline:.6f}")
    print(f"V14 독립 감사: {len(configs)}개 설정 × {len(AUDIT_SEEDS)}개 새 시드")
    for index, (n_bins, strength) in enumerate(configs, 1):
        scores = [
            crossfit(prediction, target, seed, n_bins, strength)
            for seed in AUDIT_SEEDS
        ]
        improvements = [baseline - score for score in scores]
        rows.append(
            {
                "n_bins": n_bins,
                "strength": strength,
                "mean_MAE": float(np.mean(scores)),
                "worst_MAE": float(np.max(scores)),
                "std_MAE": float(np.std(scores)),
                "mean_improvement": float(np.mean(improvements)),
                "worst_improvement": float(np.min(improvements)),
                "wins": int(sum(value > 0 for value in improvements)),
                **{f"audit_seed_{seed}": score for seed, score in zip(AUDIT_SEEDS, scores)},
            }
        )
        print(
            f"[{index:02d}/{len(configs):02d}] bins={n_bins:<2} strength={strength:.2f} "
            f"평균={np.mean(scores):.6f} | 최악 개선={np.min(improvements):+.6f} | "
            f"승리={sum(value > 0 for value in improvements)}/{len(AUDIT_SEEDS)}",
            flush=True,
        )

    result = pd.DataFrame(rows).sort_values(
        ["mean_MAE", "worst_improvement", "std_MAE"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    path = RESULT_DIR / "svr_v14_calibration_audit.csv"
    result.to_csv(path, index=False)
    best = result.iloc[0]

    print("\n================ SVR V14 감사 결과 ================")
    print(f"기준 OOF MAE    : {baseline:.6f}")
    print(f"최고 구간 수    : {int(best['n_bins'])}")
    print(f"보정 강도       : {best['strength']:.2f}")
    print(f"평균 Audit MAE  : {best['mean_MAE']:.6f}")
    print(f"평균 개선       : {best['mean_improvement']:+.6f}")
    print(f"최악 개선       : {best['worst_improvement']:+.6f}")
    print(f"개선 분할       : {int(best['wins'])}/{len(AUDIT_SEEDS)}")
    print("====================================================")
    print(f"Saved: {path.resolve()}")


if __name__ == "__main__":
    main()
