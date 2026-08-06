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
V19_BY_SEED = {
    42: 0.1575238167164372,
    77: 0.1564966653883252,
    2026: 0.1547984148591265,
}
C = 5.0
CAT_WEIGHT = 5.0

ORIGINAL_NUMERIC = {
    "age",
    "height",
    "weight",
    "cholesterol",
    "systolic_blood_pressure",
    "diastolic_blood_pressure",
    "glucose",
    "bone_density",
    "mean_working",
}
CORE_PHYSIOLOGY = {
    "age",
    "bmi",
    "mean_arterial_pressure",
    "pulse_pressure",
    "glucose",
    "cholesterol",
    "bone_density",
    "allostatic_count",
    "mean_working",
}


def select_variant(full: pd.DataFrame, variant: str) -> pd.DataFrame:
    categorical = full.select_dtypes(exclude="number").columns.tolist()
    numerical = full.select_dtypes(include="number").columns.tolist()
    if variant == "full":
        return full.copy()
    if variant == "no_numeric_copies":
        drop = [column for column in full if "__weight_copy_" in column]
        return full.drop(columns=drop)
    if variant == "categorical_plus_original":
        keep = categorical + [column for column in numerical if column in ORIGINAL_NUMERIC]
        return full[keep].copy()
    if variant == "categorical_plus_core":
        keep = categorical + [column for column in numerical if column in CORE_PHYSIOLOGY]
        return full[keep].copy()
    if variant == "categorical_only":
        return full[categorical].copy()
    if variant == "numeric_only":
        return full[numerical].copy()
    raise ValueError(variant)


def cv_score(features, target, seed, gamma):
    oof = np.zeros(len(features), dtype=float)
    folds = []
    for train_idx, valid_idx in KFold(5, shuffle=True, random_state=seed).split(features):
        model = make_block_weighted_model(features, C, gamma, CAT_WEIGHT)
        model.fit(features.iloc[train_idx], target[train_idx])
        pred = np.clip(model.predict(features.iloc[valid_idx]), 0.0, 1.0)
        oof[valid_idx] = pred
        folds.append(mean_absolute_error(target[valid_idx], pred))
    return mean_absolute_error(target, oof), folds


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    full = build_v10_winner(train.drop(columns=[ID_COLUMN, TARGET]))
    target = train[TARGET].to_numpy(float)

    variants = (
        "full",
        "no_numeric_copies",
        "categorical_plus_original",
        "categorical_plus_core",
        "categorical_only",
        "numeric_only",
    )
    configs = [
        (variant, gamma)
        for variant in variants
        for gamma in (0.015, 0.025, 0.035, 0.045, 0.060, 0.080)
    ]
    rows = []
    print(f"Stage 1: 고범주 SVR 숫자 피처 제거 후보 {len(configs)}개를 탐색합니다.")
    for index, (variant, gamma) in enumerate(configs, 1):
        features = select_variant(full, variant)
        score, folds = cv_score(features, target, 42, gamma)
        rows.append(
            {
                "variant": variant,
                "C": C,
                "categorical_weight": CAT_WEIGHT,
                "gamma": gamma,
                "screen_MAE": score,
                "fold_std": float(np.std(folds)),
                "n_features": features.shape[1],
                "n_numeric": features.select_dtypes(include="number").shape[1],
                "n_categorical": features.select_dtypes(exclude="number").shape[1],
            }
        )
        print(
            f"[{index:02d}/{len(configs):02d}] {variant:<27} "
            f"features={features.shape[1]:<2} gamma={gamma:<5} MAE={score:.6f}",
            flush=True,
        )

    screen = pd.DataFrame(rows).sort_values(["screen_MAE", "fold_std"])
    screen_path = RESULT_DIR / "svr_v21_stage1_highcat_numeric_ablation.csv"
    screen.to_csv(screen_path, index=False)
    finalists = screen.groupby("variant", group_keys=False).head(1).sort_values("screen_MAE")

    confirmed = []
    print("\nStage 2: 구성별 최고 후보를 3개 시드에서 V19와 비교합니다.")
    for index, row in enumerate(finalists.itertuples(index=False), 1):
        features = select_variant(full, row.variant)
        scores, all_folds, improvements = [], [], []
        for seed in SEEDS:
            score, folds = cv_score(features, target, seed, float(row.gamma))
            scores.append(score)
            all_folds.extend(folds)
            improvements.append(V19_BY_SEED[seed] - score)
        confirmed.append(
            {
                "variant": row.variant,
                "C": C,
                "categorical_weight": CAT_WEIGHT,
                "gamma": row.gamma,
                "n_features": row.n_features,
                "mean_MAE": float(np.mean(scores)),
                "worst_seed_MAE": float(np.max(scores)),
                "fold_std": float(np.std(all_folds)),
                "seed_42": scores[0],
                "seed_77": scores[1],
                "seed_2026": scores[2],
                "mean_improvement_vs_v19": float(np.mean(improvements)),
                "worst_improvement_vs_v19": float(np.min(improvements)),
                "wins_vs_v19": int(sum(value > 0 for value in improvements)),
            }
        )
        print(
            f"[Confirm {index}/{len(finalists)}] {row.variant:<27} "
            f"평균={np.mean(scores):.6f} | 최악 개선={np.min(improvements):+.6f} | "
            f"V19 승리={sum(value > 0 for value in improvements)}/3",
            flush=True,
        )

    result = pd.DataFrame(confirmed).sort_values(
        ["mean_MAE", "worst_improvement_vs_v19", "fold_std"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    result_path = RESULT_DIR / "svr_v21_stage2_highcat_numeric_confirm.csv"
    result.to_csv(result_path, index=False)
    best = result.iloc[0]

    print("\n================ SVR V21 최종 결과 ================")
    print(f"최고 피처 구성   : {best['variant']}")
    print(f"피처 수          : {int(best['n_features'])}")
    print(f"최고 설정        : C={C}, cat_weight={CAT_WEIGHT}, gamma={best['gamma']}")
    print(f"평균 CV MAE      : {best['mean_MAE']:.6f}")
    print(f"최악 시드 MAE    : {best['worst_seed_MAE']:.6f}")
    print(f"V19 대비 개선    : {best['mean_improvement_vs_v19']:+.6f}")
    print(f"최악 시드 개선   : {best['worst_improvement_vs_v19']:+.6f}")
    print(f"V19 승리 시드    : {int(best['wins_vs_v19'])}/3")
    print("====================================================")
    print(f"Screen results : {screen_path.resolve()}")
    print(f"Confirm results: {result_path.resolve()}")


if __name__ == "__main__":
    main()
