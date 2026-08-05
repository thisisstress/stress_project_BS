from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


DATA_DIR = Path("/Users/bitsaem/Desktop/stress_project/open")
RESULT_DIR = Path("/Users/bitsaem/Desktop/stress_project/8/5/8/5_result")
OOF_PATH = RESULT_DIR / "svr_v11_best_oof.csv"
META_SEEDS = (314, 728, 1024, 2026, 7777)


def fit_predict(method: str, train_pred, train_y, valid_pred):
    train_pred = np.asarray(train_pred, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    valid_pred = np.asarray(valid_pred, dtype=float)

    if method == "identity":
        return valid_pred

    if method == "median_shift":
        shift = float(np.median(train_y - train_pred))
        return valid_pred + shift

    if method.startswith("shrink_"):
        factor = float(method.split("_")[1])
        center = float(np.median(train_pred))
        transformed_train = center + factor * (train_pred - center)
        shift = float(np.median(train_y - transformed_train))
        return center + factor * (valid_pred - center) + shift

    if method.startswith("quantile_linear_"):
        alpha = float(method.rsplit("_", 1)[1])
        model = QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs")
        model.fit(train_pred.reshape(-1, 1), train_y)
        return model.predict(valid_pred.reshape(-1, 1))

    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(train_pred, train_y)
        return model.predict(valid_pred)

    if method.startswith("bin_shift_"):
        n_bins = int(method.rsplit("_", 1)[1])
        edges = np.unique(np.quantile(train_pred, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:
            return valid_pred + np.median(train_y - train_pred)
        edges[0], edges[-1] = -np.inf, np.inf
        train_bins = np.digitize(train_pred, edges[1:-1], right=True)
        valid_bins = np.digitize(valid_pred, edges[1:-1], right=True)
        global_shift = float(np.median(train_y - train_pred))
        shifts = {}
        for bin_id in range(len(edges) - 1):
            mask = train_bins == bin_id
            shifts[bin_id] = (
                float(np.median(train_y[mask] - train_pred[mask]))
                if mask.sum() >= 30
                else global_shift
            )
        return valid_pred + np.array([shifts[value] for value in valid_bins])

    raise ValueError(f"Unknown method: {method}")


def crossfit(method, prediction, target, seed):
    calibrated = np.zeros(len(target), dtype=float)
    splitter = KFold(5, shuffle=True, random_state=seed)
    for train_idx, valid_idx in splitter.split(prediction):
        calibrated[valid_idx] = fit_predict(
            method,
            prediction[train_idx],
            target[train_idx],
            prediction[valid_idx],
        )
    calibrated = np.clip(calibrated, 0.0, 1.0)
    return mean_absolute_error(target, calibrated), calibrated


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    oof = pd.read_csv(OOF_PATH)
    prediction = oof["pred_mean"].to_numpy(float)
    target = oof["stress_score"].to_numpy(float)
    baseline = mean_absolute_error(target, prediction)

    methods = (
        "identity",
        "median_shift",
        "shrink_0.90",
        "shrink_0.95",
        "shrink_0.98",
        "shrink_1.02",
        "shrink_1.05",
        "shrink_1.10",
        "quantile_linear_0.0",
        "quantile_linear_0.0001",
        "quantile_linear_0.001",
        "isotonic",
        "bin_shift_5",
        "bin_shift_10",
        "bin_shift_15",
    )

    rows = []
    print(f"V11 3-seed OOF 평균 baseline MAE: {baseline:.6f}")
    print(f"보정 후보 {len(methods)}개를 meta seed {len(META_SEEDS)}개로 교차검증합니다.")
    for index, method in enumerate(methods, 1):
        scores = []
        improvements = []
        for seed in META_SEEDS:
            score, _ = crossfit(method, prediction, target, seed)
            scores.append(score)
            improvements.append(baseline - score)
        rows.append(
            {
                "method": method,
                "mean_MAE": float(np.mean(scores)),
                "worst_meta_MAE": float(np.max(scores)),
                "mean_improvement": float(np.mean(improvements)),
                "worst_improvement": float(np.min(improvements)),
                "wins": int(sum(value > 0 for value in improvements)),
                **{f"meta_seed_{seed}": score for seed, score in zip(META_SEEDS, scores)},
            }
        )
        print(
            f"[{index:02d}/{len(methods):02d}] {method:<24} "
            f"평균={np.mean(scores):.6f} | 최악 개선={np.min(improvements):+.6f} | "
            f"승리={sum(value > 0 for value in improvements)}/{len(META_SEEDS)}",
            flush=True,
        )

    result = pd.DataFrame(rows).sort_values(
        ["mean_MAE", "worst_improvement"], ascending=[True, False]
    ).reset_index(drop=True)
    result_path = RESULT_DIR / "svr_v13_oof_calibration_results.csv"
    result.to_csv(result_path, index=False)
    best = result.iloc[0]

    # 진단용 교차적합 예측: 고정 meta seed에서 각 행의 정답을 제외하고 보정한다.
    _, calibrated = crossfit(best["method"], prediction, target, META_SEEDS[0])
    diagnostic = oof[["ID", "stress_score", "pred_mean"]].copy()
    diagnostic["pred_calibrated"] = calibrated
    diagnostic["abs_error_before"] = np.abs(target - prediction)
    diagnostic["abs_error_after"] = np.abs(target - calibrated)
    diagnostic_path = RESULT_DIR / "svr_v13_best_crossfit_oof.csv"
    diagnostic.to_csv(diagnostic_path, index=False)

    print("\n================ SVR V13 최종 결과 ================")
    print(f"기준 OOF MAE    : {baseline:.6f}")
    print(f"최고 보정법     : {best['method']}")
    print(f"평균 Meta MAE   : {best['mean_MAE']:.6f}")
    print(f"평균 개선       : {best['mean_improvement']:+.6f}")
    print(f"최악 개선       : {best['worst_improvement']:+.6f}")
    print(f"개선 분할       : {int(best['wins'])}/{len(META_SEEDS)}")
    print("====================================================")
    print(f"Results: {result_path.resolve()}")
    print(f"OOF    : {diagnostic_path.resolve()}")


if __name__ == "__main__":
    main()
