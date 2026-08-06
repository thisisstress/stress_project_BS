from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from experiment_svr_v10_kernel_tuning_v11 import build_v10_winner


DATA_DIR = Path("/Users/bitsaem/Desktop/stress_project/open")
RESULT_DIR = Path("/Users/bitsaem/Desktop/stress_project/8/5/8/5_result")
TEAM_NOTEBOOK = Path(
    "/Users/bitsaem/Desktop/stress_project/stress_prediction_co/"
    "stress_prediction_team_best_v7_verified.ipynb"
)
TARGET = "stress_score"
ID_COLUMN = "ID"
SEEDS = (42, 77, 2026)
BASE_BY_SEED = {
    42: 0.14855,
    77: 0.14638666666666666,
    2026: 0.14473333333333332,
}

PHYSIOLOGY = {
    "metabolic_product", "risk_bmi_25", "risk_bmi_30", "risk_bp_140_90",
    "risk_glucose_99", "risk_cholesterol_250", "allostatic_count",
    "age_allostatic", "age_bmi", "age_map", "age_glucose",
    "age_bone_density",
}
HISTORY = {
    "personal_family_pair", "same_history", "history_count",
    "age_history_count", "allostatic_history",
}
ORDINAL = {
    "education_level_numeric", "education_missing", "activity_intensity",
    "activity_distance_moderate", "smoking_exposure", "current_smoker_flag",
}
LIFESTYLE = {
    "education_activity_pair", "education_smoking_pair", "activity_smoking_pair",
    "education_activity_smoking_group", "education_x_activity",
    "education_x_smoking", "activity_x_smoking", "ordinal_lifestyle_sum",
    "ordinal_lifestyle_product",
}
SEX = {
    "female_x_bmi", "male_x_bmi", "female_x_mean_arterial_pressure",
    "male_x_mean_arterial_pressure", "female_x_pulse_pressure",
    "male_x_pulse_pressure", "female_x_glucose", "male_x_glucose",
    "female_x_cholesterol", "male_x_cholesterol", "female_x_bone_density",
    "male_x_bone_density", "female_x_allostatic_count",
    "male_x_allostatic_count",
}


def load_team_functions():
    notebook = json.loads(TEAM_NOTEBOOK.read_text(encoding="utf-8"))
    code = "".join(notebook["cells"][1]["source"])
    namespace = {}
    exec(compile(code, str(TEAM_NOTEBOOK), "exec"), namespace)
    return (
        namespace["add_features"],
        namespace["tree_quantile_prediction"],
        namespace["pair_neighbor_prediction"],
        float(namespace["PAIR_WEIGHT"]),
    )


def build_variant(raw, variant, add_features):
    base = add_features(raw).reset_index(drop=True)
    if variant == "baseline":
        return base
    full = build_v10_winner(raw).reset_index(drop=True)
    groups = {
        "physiology": PHYSIOLOGY,
        "history": PHYSIOLOGY | HISTORY,
        "ordinal": PHYSIOLOGY | HISTORY | ORDINAL,
        "lifestyle": PHYSIOLOGY | HISTORY | ORDINAL | LIFESTYLE,
        "sex": PHYSIOLOGY | HISTORY | ORDINAL | SEX,
        "lifestyle_sex": PHYSIOLOGY | HISTORY | ORDINAL | LIFESTYLE | SEX,
    }
    if variant == "full_v10":
        selected = [column for column in full.columns if "__weight_copy_" not in column]
    else:
        selected = [column for column in groups[variant] if column in full.columns]
    result = base.copy()
    for column in selected:
        if column not in result.columns:
            result[column] = full[column]
    return result


def cv_score(features, target, seed, tree_predict, pair_predict, pair_weight):
    oof = np.zeros(len(features), dtype=float)
    folds = []
    splitter = KFold(5, shuffle=True, random_state=seed)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(features), 1):
        train_x = features.iloc[train_idx]
        valid_x = features.iloc[valid_idx]
        train_y = target.iloc[train_idx]
        tree = tree_predict(train_x, valid_x, train_y)
        pair = pair_predict(train_x, valid_x, train_y)
        prediction = np.clip(
            np.round((1.0 - pair_weight) * tree + pair_weight * pair, 2),
            0.0,
            1.0,
        )
        oof[valid_idx] = prediction
        folds.append(mean_absolute_error(target.iloc[valid_idx], prediction))
        print(f"Seed {seed} | Fold {fold}/5 complete", flush=True)
    return mean_absolute_error(target, oof), folds


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    add_features, tree_predict, pair_predict, pair_weight = load_team_functions()
    train = pd.read_csv(DATA_DIR / "train.csv")
    raw = train.drop(columns=[ID_COLUMN, TARGET])
    target = train[TARGET].reset_index(drop=True)
    variants = (
        "baseline", "physiology", "history", "ordinal", "lifestyle",
        "sex", "lifestyle_sex", "full_v10",
    )

    screen_rows = []
    print("Stage 1: SVR 피처군을 팀 최고 모델에 이식해 seed=42에서 비교합니다.")
    for index, variant in enumerate(variants, 1):
        features = build_variant(raw, variant, add_features)
        score, folds = cv_score(
            features, target, 42, tree_predict, pair_predict, pair_weight
        )
        screen_rows.append({
            "variant": variant,
            "feature_count": features.shape[1],
            "screen_MAE": score,
            "fold_std": float(np.std(folds)),
            "improvement_vs_base42": BASE_BY_SEED[42] - score,
        })
        print(
            f"[{index}/{len(variants)}] {variant:<14} features={features.shape[1]:<3} "
            f"MAE={score:.6f} | 개선={BASE_BY_SEED[42]-score:+.6f}",
            flush=True,
        )

    screen = pd.DataFrame(screen_rows).sort_values(["screen_MAE", "fold_std"])
    screen_path = RESULT_DIR / "team_v35_stage1_feature_transfer.csv"
    screen.to_csv(screen_path, index=False)
    finalists = screen.head(4)

    rows = []
    print("\nStage 2: 상위 4개를 동일한 3개 시드에서 팀 기준과 비교합니다.")
    for row in finalists.itertuples(index=False):
        features = build_variant(raw, row.variant, add_features)
        scores, all_folds, improvements = [], [], []
        for seed in SEEDS:
            score, folds = cv_score(
                features, target, seed, tree_predict, pair_predict, pair_weight
            )
            scores.append(score)
            all_folds.extend(folds)
            improvements.append(BASE_BY_SEED[seed] - score)
        rows.append({
            "variant": row.variant,
            "feature_count": row.feature_count,
            "mean_MAE": float(np.mean(scores)),
            "worst_seed_MAE": float(np.max(scores)),
            "fold_std": float(np.std(all_folds)),
            "seed_42": scores[0],
            "seed_77": scores[1],
            "seed_2026": scores[2],
            "mean_improvement": float(np.mean(improvements)),
            "worst_improvement": float(np.min(improvements)),
            "wins": int(sum(value > 0 for value in improvements)),
        })
        print(
            f"[Confirm] {row.variant:<14} 평균={np.mean(scores):.6f} | "
            f"최악 개선={np.min(improvements):+.6f} | "
            f"승리={sum(value > 0 for value in improvements)}/3",
            flush=True,
        )

    result = pd.DataFrame(rows).sort_values(
        ["wins", "worst_improvement", "mean_MAE"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    result_path = RESULT_DIR / "team_v35_stage2_feature_transfer_confirm.csv"
    result.to_csv(result_path, index=False)
    best = result.iloc[0]
    print("\n================ Team V35 최종 결과 ================")
    print(f"최고 피처군      : {best.variant}")
    print(f"평균 CV MAE      : {best.mean_MAE:.6f}")
    print(f"최악 시드 MAE    : {best.worst_seed_MAE:.6f}")
    print(f"기준 대비 개선   : {best.mean_improvement:+.6f}")
    print(f"최악 시드 개선   : {best.worst_improvement:+.6f}")
    print(f"기준 승리 시드   : {int(best.wins)}/3")
    print("=====================================================")
    print(f"Screen : {Path(screen_path).resolve()}")
    print(f"Confirm: {Path(result_path).resolve()}")


if __name__ == "__main__":
    main()
