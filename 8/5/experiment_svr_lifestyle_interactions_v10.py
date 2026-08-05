from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from experiment_svr_demographic_interactions_v6 import add_interactions, make_model
from experiment_svr_literature_features import add_literature_features
from experiment_svr_positive_ordinal_v9 import build_variant


DATA_DIR = Path("/Users/bitsaem/Desktop/stress_project/open")
RESULT_DIR = Path("/Users/bitsaem/Desktop/stress_project/8/5/8/5_result")
TARGET = "stress_score"
ID_COLUMN = "ID"
SEEDS = (42, 77, 2026)
V9_BY_SEED = {42: 0.166195, 77: 0.165007, 2026: 0.164704}


def add_v10_features(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    x = frame.copy()
    education = x["edu_level"].fillna("missing").astype(str)
    activity = x["activity"].fillna("missing").astype(str)
    smoking = x["smoke_status"].fillna("missing").astype(str)

    if variant in {"pair_categories", "all_lifestyle"}:
        x["education_activity_pair"] = education + "__" + activity
        x["education_smoking_pair"] = education + "__" + smoking
        x["activity_smoking_pair"] = activity + "__" + smoking

    if variant in {"triple_category", "all_lifestyle"}:
        x["education_activity_smoking_group"] = (
            education + "__" + activity + "__" + smoking
        )

    if variant in {"numeric_products", "all_lifestyle"}:
        e = x["education_level_numeric"].fillna(-1.0)
        a = x["activity_intensity"].fillna(1.0)
        s = x["smoking_exposure"].fillna(0.0)
        x["education_x_activity"] = e * a
        x["education_x_smoking"] = e * s
        x["activity_x_smoking"] = a * s
        x["ordinal_lifestyle_sum"] = e + a - s
        x["ordinal_lifestyle_product"] = (e + 1.0) * (a + 1.0) / (s + 1.0)

    if variant in {"sex_physiology", "lifestyle_plus_sex"}:
        female = (x["gender"] == "F").astype(float)
        male = (x["gender"] == "M").astype(float)
        for column in (
            "bmi",
            "mean_arterial_pressure",
            "pulse_pressure",
            "glucose",
            "cholesterol",
            "bone_density",
            "allostatic_count",
        ):
            x[f"female_x_{column}"] = female * x[column]
            x[f"male_x_{column}"] = male * x[column]

    if variant == "lifestyle_plus_sex":
        x = add_v10_features(x, "all_lifestyle")

    return x


def cv_score(features, target, seed, c, gamma):
    oof = np.zeros(len(features))
    folds = []
    for train_idx, valid_idx in KFold(5, shuffle=True, random_state=seed).split(features):
        model = make_model(features, c, gamma)
        model.fit(features.iloc[train_idx], target[train_idx])
        prediction = np.clip(model.predict(features.iloc[valid_idx]), 0.0, 1.0)
        oof[valid_idx] = prediction
        folds.append(mean_absolute_error(target[valid_idx], prediction))
    return mean_absolute_error(target, oof), folds


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    physiology = add_literature_features(
        train.drop(columns=[ID_COLUMN, TARGET]), "physiology"
    )
    v6_base = add_interactions(physiology, "history")
    v9_base = build_variant(v6_base, "education_activity_smoking")
    target = train[TARGET].to_numpy(float)

    variants = (
        "baseline",
        "pair_categories",
        "triple_category",
        "numeric_products",
        "all_lifestyle",
        "sex_physiology",
        "lifestyle_plus_sex",
    )
    configs = [
        (variant, c, gamma)
        for variant in variants
        for c in (0.7, 1.0, 1.5)
        for gamma in (0.08, 0.10, 0.12)
    ]
    screen_rows = []
    print(f"Stage 1: 생활습관 상호작용 후보 {len(configs)}개를 탐색합니다.")
    for index, (variant, c, gamma) in enumerate(configs, 1):
        features = add_v10_features(v9_base, "none" if variant == "baseline" else variant)
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
            f"[{index:02d}/{len(configs):02d}] {variant:<20} C={c} gamma={gamma} "
            f"MAE={score:.6f}",
            flush=True,
        )

    screen = pd.DataFrame(screen_rows).sort_values(["screen_MAE", "fold_std"])
    screen.to_csv(RESULT_DIR / "svr_v10_stage1_lifestyle_screen.csv", index=False)
    finalists = screen.groupby("variant", group_keys=False).head(1).sort_values("screen_MAE")

    rows = []
    print("\nStage 2: 각 상호작용 최고 후보를 3개 시드로 확인합니다.")
    for index, row in enumerate(finalists.itertuples(index=False), 1):
        features = add_v10_features(v9_base, "none" if row.variant == "baseline" else row.variant)
        seed_scores = []
        fold_scores = []
        improvements = []
        for seed in SEEDS:
            score, folds = cv_score(features, target, seed, float(row.C), float(row.gamma))
            seed_scores.append(score)
            fold_scores.extend(folds)
            improvements.append(V9_BY_SEED[seed] - score)
        rows.append(
            {
                "variant": row.variant,
                "C": row.C,
                "gamma": row.gamma,
                "mean_MAE": float(np.mean(seed_scores)),
                "worst_seed_MAE": float(np.max(seed_scores)),
                "fold_std": float(np.std(fold_scores)),
                "seed_42": seed_scores[0],
                "seed_77": seed_scores[1],
                "seed_2026": seed_scores[2],
                "mean_improvement_vs_v9": float(np.mean(improvements)),
                "worst_improvement_vs_v9": float(np.min(improvements)),
                "wins_vs_v9": int(sum(value > 0 for value in improvements)),
            }
        )
        print(
            f"[Confirm {index}/{len(finalists)}] {row.variant:<20} "
            f"평균={np.mean(seed_scores):.6f} | 개선={np.mean(improvements):+.6f} | "
            f"V9 승리={sum(value > 0 for value in improvements)}/3",
            flush=True,
        )

    result = pd.DataFrame(rows).sort_values(
        ["mean_MAE", "worst_improvement_vs_v9", "fold_std"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    path = RESULT_DIR / "svr_v10_stage2_lifestyle_confirm.csv"
    result.to_csv(path, index=False)
    best = result.iloc[0]
    print("\n=============== SVR V10 최종 결과 ===============")
    print(f"최고 피처군   : {best['variant']}")
    print(f"최고 설정     : C={best['C']}, gamma={best['gamma']}, epsilon=0")
    print(f"평균 CV MAE   : {best['mean_MAE']:.6f}")
    print(f"최악 시드 MAE : {best['worst_seed_MAE']:.6f}")
    print(f"V9 대비 개선  : {best['mean_improvement_vs_v9']:+.6f}")
    print(f"최악 개선량   : {best['worst_improvement_vs_v9']:+.6f}")
    print(f"V9 승리 시드  : {int(best['wins_vs_v9'])}/3")
    print("===================================================")
    print(f"Saved: {path.resolve()}")


if __name__ == "__main__":
    main()
