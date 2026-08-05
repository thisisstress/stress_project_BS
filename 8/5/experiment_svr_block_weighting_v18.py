from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from experiment_svr_block_weighting_v17 import make_block_weighted_model
from experiment_svr_v10_kernel_tuning_v11 import build_v10_winner


DATA_DIR = Path("/Users/bitsaem/Desktop/stress_project/open")
RESULT_DIR = Path("/Users/bitsaem/Desktop/stress_project/8/5/8/5_result")
TARGET = "stress_score"
ID_COLUMN = "ID"
SEEDS = (42, 77, 2026)
V17_BY_SEED = {
    42: 0.1618743543497014,
    77: 0.1601074887284556,
    2026: 0.1594996809010961,
}
C = 5.0


def cv_score(features, target, seed, gamma, categorical_weight):
    oof = np.zeros(len(features), dtype=float)
    folds = []
    for train_idx, valid_idx in KFold(5, shuffle=True, random_state=seed).split(features):
        model = make_block_weighted_model(features, C, gamma, categorical_weight)
        model.fit(features.iloc[train_idx], target[train_idx])
        pred = np.clip(model.predict(features.iloc[valid_idx]), 0.0, 1.0)
        oof[valid_idx] = pred
        folds.append(mean_absolute_error(target[valid_idx], pred))
    return mean_absolute_error(target, oof), folds


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    features = build_v10_winner(train.drop(columns=[ID_COLUMN, TARGET]))
    target = train[TARGET].to_numpy(float)

    weights = (1.50, 1.75, 2.00, 2.25, 2.50, 3.00)
    gammas = (0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10)
    configs = [(weight, gamma) for weight in weights for gamma in gammas]
    rows = []
    print(f"Stage 1: 확장 범주형 거리 후보 {len(configs)}개를 seed=42에서 탐색합니다.")
    for index, (weight, gamma) in enumerate(configs, 1):
        score, folds = cv_score(features, target, 42, gamma, weight)
        rows.append(
            {
                "categorical_weight": weight,
                "C": C,
                "gamma": gamma,
                "screen_MAE": score,
                "fold_std": float(np.std(folds)),
            }
        )
        print(
            f"[{index:02d}/{len(configs):02d}] cat_weight={weight:<4} "
            f"gamma={gamma:<4} MAE={score:.6f}",
            flush=True,
        )

    screen = pd.DataFrame(rows).sort_values(["screen_MAE", "fold_std"])
    screen_path = RESULT_DIR / "svr_v18_stage1_extended_block_weighting.csv"
    screen.to_csv(screen_path, index=False)
    finalists = screen.head(10)

    confirmed = []
    print("\nStage 2: 상위 10개 후보를 3개 시드에서 V17과 비교합니다.")
    for index, row in enumerate(finalists.itertuples(index=False), 1):
        scores, all_folds, improvements = [], [], []
        for seed in SEEDS:
            score, folds = cv_score(
                features,
                target,
                seed,
                float(row.gamma),
                float(row.categorical_weight),
            )
            scores.append(score)
            all_folds.extend(folds)
            improvements.append(V17_BY_SEED[seed] - score)
        confirmed.append(
            {
                "categorical_weight": row.categorical_weight,
                "C": C,
                "gamma": row.gamma,
                "mean_MAE": float(np.mean(scores)),
                "worst_seed_MAE": float(np.max(scores)),
                "fold_std": float(np.std(all_folds)),
                "seed_42": scores[0],
                "seed_77": scores[1],
                "seed_2026": scores[2],
                "mean_improvement_vs_v17": float(np.mean(improvements)),
                "worst_improvement_vs_v17": float(np.min(improvements)),
                "wins_vs_v17": int(sum(value > 0 for value in improvements)),
            }
        )
        print(
            f"[Confirm {index:02d}/{len(finalists):02d}] weight={row.categorical_weight:<4} "
            f"gamma={row.gamma:<4} 평균={np.mean(scores):.6f} | "
            f"최악 개선={np.min(improvements):+.6f} | "
            f"V17 승리={sum(value > 0 for value in improvements)}/3",
            flush=True,
        )

    result = pd.DataFrame(confirmed).sort_values(
        ["mean_MAE", "worst_improvement_vs_v17", "fold_std"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    result_path = RESULT_DIR / "svr_v18_stage2_extended_block_confirm.csv"
    result.to_csv(result_path, index=False)
    best = result.iloc[0]

    print("\n================ SVR V18 최종 결과 ================")
    print(f"범주형 거리 가중치: {best['categorical_weight']}")
    print(f"최고 설정         : C={C}, gamma={best['gamma']}, epsilon=0")
    print(f"평균 CV MAE       : {best['mean_MAE']:.6f}")
    print(f"최악 시드 MAE     : {best['worst_seed_MAE']:.6f}")
    print(f"V17 대비 개선     : {best['mean_improvement_vs_v17']:+.6f}")
    print(f"최악 시드 개선    : {best['worst_improvement_vs_v17']:+.6f}")
    print(f"V17 승리 시드     : {int(best['wins_vs_v17'])}/3")
    print("====================================================")
    print(f"Screen results : {screen_path.resolve()}")
    print(f"Confirm results: {result_path.resolve()}")


if __name__ == "__main__":
    main()
