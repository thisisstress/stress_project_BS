from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from experiment_svr_demographic_interactions_v6 import make_model
from experiment_svr_v10_kernel_tuning_v11 import build_v10_winner


DATA_DIR = Path("/Users/bitsaem/Desktop/stress_project/open")
RESULT_DIR = Path("/Users/bitsaem/Desktop/stress_project/8/5/8/5_result")
TARGET = "stress_score"
ID_COLUMN = "ID"
SEEDS = (42, 77, 2026)
V11_BY_SEED = {
    42: 0.1632624252055463,
    77: 0.1619455266289465,
    2026: 0.1613116012321137,
}

HISTORY_COLUMNS = {
    "personal_family_pair",
    "same_history",
    "history_count",
    "age_history_count",
    "allostatic_history",
}
ORDINAL_COLUMNS = {
    "education_level_numeric",
    "education_missing",
    "activity_intensity",
    "activity_distance_moderate",
    "smoking_exposure",
    "current_smoker_flag",
}
PAIR_COLUMNS = {
    "education_activity_pair",
    "education_smoking_pair",
    "activity_smoking_pair",
    "education_activity_smoking_group",
}
NUMERIC_LIFESTYLE_COLUMNS = {
    "education_x_activity",
    "education_x_smoking",
    "activity_x_smoking",
    "ordinal_lifestyle_sum",
    "ordinal_lifestyle_product",
}
RAW_LIFESTYLE_COLUMNS = {"edu_level", "activity", "smoke_status"}


def select_variant(full: pd.DataFrame, variant: str) -> pd.DataFrame:
    drop = set()
    if variant == "no_weight_copies":
        drop |= {column for column in full if "__weight_copy_" in column}
    elif variant == "no_history":
        drop |= HISTORY_COLUMNS
    elif variant == "no_ordinals":
        drop |= ORDINAL_COLUMNS
    elif variant == "no_pairs":
        drop |= PAIR_COLUMNS
    elif variant == "no_numeric_lifestyle":
        drop |= NUMERIC_LIFESTYLE_COLUMNS
    elif variant == "no_sex_interactions":
        drop |= {
            column
            for column in full
            if column.startswith("female_x_") or column.startswith("male_x_")
        }
    elif variant == "no_raw_lifestyle":
        drop |= RAW_LIFESTYLE_COLUMNS
    elif variant == "compact":
        drop |= {column for column in full if "__weight_copy_" in column}
        drop |= HISTORY_COLUMNS
        drop |= RAW_LIFESTYLE_COLUMNS
    elif variant != "full":
        raise ValueError(variant)
    return full.drop(columns=sorted(drop & set(full.columns)))


def cv_score(features, target, seed, c, gamma):
    oof = np.zeros(len(features), dtype=float)
    fold_scores = []
    splitter = KFold(5, shuffle=True, random_state=seed)
    for train_idx, valid_idx in splitter.split(features):
        model = make_model(features, c, gamma)
        model.fit(features.iloc[train_idx], target[train_idx])
        pred = np.clip(model.predict(features.iloc[valid_idx]), 0.0, 1.0)
        oof[valid_idx] = pred
        fold_scores.append(mean_absolute_error(target[valid_idx], pred))
    return mean_absolute_error(target, oof), fold_scores


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    full = build_v10_winner(train.drop(columns=[ID_COLUMN, TARGET]))
    target = train[TARGET].to_numpy(float)

    variants = (
        "full",
        "no_weight_copies",
        "no_history",
        "no_ordinals",
        "no_pairs",
        "no_numeric_lifestyle",
        "no_sex_interactions",
        "no_raw_lifestyle",
        "compact",
    )
    configs = [
        (variant, c, gamma)
        for variant in variants
        for c in (3.0, 5.0)
        for gamma in (0.06, 0.08)
    ]
    screen_rows = []
    print(f"Stage 1: 최고 SVR 피처 조합의 제거 후보 {len(configs)}개를 탐색합니다.")
    for index, (variant, c, gamma) in enumerate(configs, 1):
        features = select_variant(full, variant)
        score, folds = cv_score(features, target, 42, c, gamma)
        screen_rows.append(
            {
                "variant": variant,
                "C": c,
                "gamma": gamma,
                "screen_MAE": score,
                "fold_std": float(np.std(folds)),
                "n_features": features.shape[1],
            }
        )
        print(
            f"[{index:02d}/{len(configs):02d}] {variant:<22} "
            f"features={features.shape[1]:<2} C={c} gamma={gamma} MAE={score:.6f}",
            flush=True,
        )

    screen = pd.DataFrame(screen_rows).sort_values(["screen_MAE", "fold_std"])
    screen_path = RESULT_DIR / "svr_v15_stage1_feature_ablation.csv"
    screen.to_csv(screen_path, index=False)
    finalists = screen.groupby("variant", group_keys=False).head(1).sort_values("screen_MAE")

    rows = []
    print("\nStage 2: 피처군별 최고 후보를 3개 시드에서 V11과 비교합니다.")
    for index, row in enumerate(finalists.itertuples(index=False), 1):
        features = select_variant(full, row.variant)
        scores, folds_all, improvements = [], [], []
        for seed in SEEDS:
            score, folds = cv_score(features, target, seed, float(row.C), float(row.gamma))
            scores.append(score)
            folds_all.extend(folds)
            improvements.append(V11_BY_SEED[seed] - score)
        rows.append(
            {
                "variant": row.variant,
                "C": row.C,
                "gamma": row.gamma,
                "n_features": row.n_features,
                "mean_MAE": float(np.mean(scores)),
                "worst_seed_MAE": float(np.max(scores)),
                "fold_std": float(np.std(folds_all)),
                "seed_42": scores[0],
                "seed_77": scores[1],
                "seed_2026": scores[2],
                "mean_improvement_vs_v11": float(np.mean(improvements)),
                "worst_improvement_vs_v11": float(np.min(improvements)),
                "wins_vs_v11": int(sum(value > 0 for value in improvements)),
            }
        )
        print(
            f"[Confirm {index}/{len(finalists)}] {row.variant:<22} "
            f"평균={np.mean(scores):.6f} | 최악 개선={np.min(improvements):+.6f} | "
            f"V11 승리={sum(value > 0 for value in improvements)}/3",
            flush=True,
        )

    result = pd.DataFrame(rows).sort_values(
        ["mean_MAE", "worst_improvement_vs_v11", "fold_std"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    result_path = RESULT_DIR / "svr_v15_stage2_feature_ablation_confirm.csv"
    result.to_csv(result_path, index=False)
    best = result.iloc[0]

    print("\n================ SVR V15 최종 결과 ================")
    print(f"최고 피처 구성   : {best['variant']}")
    print(f"사용 피처 수     : {int(best['n_features'])}")
    print(f"최고 설정        : C={best['C']}, gamma={best['gamma']}, epsilon=0")
    print(f"평균 CV MAE      : {best['mean_MAE']:.6f}")
    print(f"최악 시드 MAE    : {best['worst_seed_MAE']:.6f}")
    print(f"V11 대비 개선    : {best['mean_improvement_vs_v11']:+.6f}")
    print(f"최악 시드 개선   : {best['worst_improvement_vs_v11']:+.6f}")
    print(f"V11 승리 시드    : {int(best['wins_vs_v11'])}/3")
    print("====================================================")
    print(f"Screen results : {screen_path.resolve()}")
    print(f"Confirm results: {result_path.resolve()}")


if __name__ == "__main__":
    main()
