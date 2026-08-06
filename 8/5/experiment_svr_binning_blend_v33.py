from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from experiment_svr_binning_strategy_v32 import (
    BIN_GROUPS,
    DATA_DIR,
    ID_COLUMN,
    RESULT_DIR,
    TARGET,
    build_v10_winner,
    make_model,
)


SEEDS = (42, 77, 2026)
V30_BY_SEED = {
    42: 0.1550647911308471,
    77: 0.1536188948902386,
    2026: 0.1524526450542613,
}
GAMMA = 0.025
BIN_WEIGHT = 1.3


def paired_oof(features, target, seed, columns):
    quantile_oof = np.zeros(len(features), dtype=float)
    kmeans_oof = np.zeros(len(features), dtype=float)
    splitter = KFold(5, shuffle=True, random_state=seed)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(features), 1):
        quantile = make_model(
            features, GAMMA, columns, 4, BIN_WEIGHT, "quantile"
        )
        kmeans = make_model(
            features, GAMMA, columns, 5, BIN_WEIGHT, "kmeans"
        )
        quantile.fit(features.iloc[train_idx], target[train_idx])
        kmeans.fit(features.iloc[train_idx], target[train_idx])
        quantile_oof[valid_idx] = np.clip(
            quantile.predict(features.iloc[valid_idx]), 0.0, 1.0
        )
        kmeans_oof[valid_idx] = np.clip(
            kmeans.predict(features.iloc[valid_idx]), 0.0, 1.0
        )
        print(f"Seed {seed} | Fold {fold}/5 complete", flush=True)
    return quantile_oof, kmeans_oof


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    features = build_v10_winner(train.drop(columns=[ID_COLUMN, TARGET]))
    target = train[TARGET].to_numpy(float)
    columns = tuple(sorted(
        (BIN_GROUPS["anthropometric"] | BIN_GROUPS["metabolic"])
        - {"age_bmi", "age_glucose"}
    ))

    predictions = {}
    for seed in SEEDS:
        predictions[seed] = paired_oof(features, target, seed, columns)

    rows = []
    for kmeans_weight in np.round(np.arange(0.0, 0.65, 0.05), 2):
        scores = []
        improvements = []
        for seed in SEEDS:
            quantile_oof, kmeans_oof = predictions[seed]
            blended = (
                (1.0 - kmeans_weight) * quantile_oof
                + kmeans_weight * kmeans_oof
            )
            score = mean_absolute_error(target, blended)
            scores.append(score)
            improvements.append(V30_BY_SEED[seed] - score)
        rows.append({
            "kmeans_weight": kmeans_weight,
            "quantile_weight": 1.0 - kmeans_weight,
            "mean_MAE": float(np.mean(scores)),
            "worst_seed_MAE": float(np.max(scores)),
            "seed_42": scores[0],
            "seed_77": scores[1],
            "seed_2026": scores[2],
            "mean_improvement_vs_v30": float(np.mean(improvements)),
            "worst_improvement_vs_v30": float(np.min(improvements)),
            "wins_vs_v30": int(sum(value > 0 for value in improvements)),
        })
        print(
            f"KMeans weight={kmeans_weight:.2f} | 평균={np.mean(scores):.6f} | "
            f"최악 개선={np.min(improvements):+.6f} | "
            f"V30 승리={sum(value > 0 for value in improvements)}/3"
        )

    result = pd.DataFrame(rows).sort_values(
        ["wins_vs_v30", "worst_improvement_vs_v30", "mean_MAE"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    path = RESULT_DIR / "svr_v33_quantile_kmeans_blend.csv"
    result.to_csv(path, index=False)
    best = result.iloc[0]

    print("\n================ SVR V33 최종 결과 ================")
    print(f"Quantile / KMeans : {best['quantile_weight']:.2f} / {best['kmeans_weight']:.2f}")
    print(f"평균 CV MAE       : {best['mean_MAE']:.6f}")
    print(f"최악 시드 MAE     : {best['worst_seed_MAE']:.6f}")
    print(f"V30 대비 개선     : {best['mean_improvement_vs_v30']:+.6f}")
    print(f"최악 시드 개선    : {best['worst_improvement_vs_v30']:+.6f}")
    print(f"V30 승리 시드     : {int(best['wins_vs_v30'])}/3")
    print("====================================================")
    print(f"Saved: {Path(path).resolve()}")


if __name__ == "__main__":
    main()
