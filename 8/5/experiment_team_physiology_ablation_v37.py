from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from experiment_team_feature_transfer_v35 import (
    DATA_DIR,
    ID_COLUMN,
    RESULT_DIR,
    SEEDS,
    TARGET,
    load_team_functions,
)
from experiment_svr_v10_kernel_tuning_v11 import build_v10_winner


GROUPS = {
    "metabolic": {"metabolic_product"},
    "risk": {
        "risk_bmi_25", "risk_bmi_30", "risk_bp_140_90",
        "risk_glucose_99", "risk_cholesterol_250",
    },
    "allostatic": {"allostatic_count", "age_allostatic"},
    "age_phys": {"age_bmi", "age_map", "age_glucose", "age_bone_density"},
}
VARIANTS = {
    "metabolic": GROUPS["metabolic"],
    "risk": GROUPS["risk"],
    "allostatic": GROUPS["allostatic"],
    "age_phys": GROUPS["age_phys"],
    "risk_allostatic": GROUPS["risk"] | GROUPS["allostatic"],
    "risk_age": GROUPS["risk"] | GROUPS["age_phys"],
    "allostatic_age": GROUPS["allostatic"] | GROUPS["age_phys"],
    "metabolic_age": GROUPS["metabolic"] | GROUPS["age_phys"],
    "full_physiology": set().union(*GROUPS.values()),
}


def make_features(raw, add_features, selected=None):
    base = add_features(raw).reset_index(drop=True)
    if not selected:
        return base
    full = build_v10_winner(raw).reset_index(drop=True)
    result = base.copy()
    for column in selected:
        if column in full.columns and column not in result.columns:
            result[column] = full[column]
    return result


def predict_oof(features, target, seed, tree_predict, pair_predict, pair_weight):
    oof = np.zeros(len(features), dtype=float)
    splitter = KFold(5, shuffle=True, random_state=seed)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(features), 1):
        train_y = target.iloc[train_idx]
        tree = tree_predict(features.iloc[train_idx], features.iloc[valid_idx], train_y)
        pair = pair_predict(features.iloc[train_idx], features.iloc[valid_idx], train_y)
        oof[valid_idx] = np.clip(
            np.round((1 - pair_weight) * tree + pair_weight * pair, 2), 0, 1
        )
        print(f"Seed {seed} | Fold {fold}/5 complete", flush=True)
    return oof


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    add_features, tree_predict, pair_predict, pair_weight = load_team_functions()
    train = pd.read_csv(DATA_DIR / "train.csv")
    raw = train.drop(columns=[ID_COLUMN, TARGET])
    target = train[TARGET].reset_index(drop=True)

    print("Stage 1: 생리 피처 묶음을 seed=42에서 분해 탐색합니다.")
    base_features = make_features(raw, add_features)
    base42 = predict_oof(
        base_features, target, 42, tree_predict, pair_predict, pair_weight
    )
    base42_mae = mean_absolute_error(target, base42)
    screen_rows = []
    screen_oof = {}
    for index, (name, columns) in enumerate(VARIANTS.items(), 1):
        features = make_features(raw, add_features, columns)
        oof = predict_oof(features, target, 42, tree_predict, pair_predict, pair_weight)
        score = mean_absolute_error(target, oof)
        screen_oof[name] = oof
        screen_rows.append({
            "variant": name,
            "added_features": len(columns),
            "feature_count": features.shape[1],
            "seed42_MAE": score,
            "improvement": base42_mae - score,
        })
        print(
            f"[{index}/{len(VARIANTS)}] {name:<18} MAE={score:.6f} | "
            f"개선={base42_mae-score:+.6f}", flush=True
        )

    screen = pd.DataFrame(screen_rows).sort_values("seed42_MAE").reset_index(drop=True)
    screen_path = RESULT_DIR / "team_v37_stage1_physiology_ablation.csv"
    screen.to_csv(screen_path, index=False)
    finalists = list(dict.fromkeys(screen.head(4).variant.tolist() + ["full_physiology"]))

    print("\nStage 2: 상위 피처군의 3시드 OOF와 혼합 비율을 검증합니다.")
    base_by_seed = {42: base42}
    for seed in SEEDS[1:]:
        base_by_seed[seed] = predict_oof(
            base_features, target, seed, tree_predict, pair_predict, pair_weight
        )

    candidate_oof = {}
    for name in finalists:
        features = make_features(raw, add_features, VARIANTS[name])
        candidate_oof[name] = {42: screen_oof[name]}
        for seed in SEEDS[1:]:
            candidate_oof[name][seed] = predict_oof(
                features, target, seed, tree_predict, pair_predict, pair_weight
            )

    rows = []
    weights = np.round(np.arange(0.10, 1.01, 0.05), 2)
    for name in finalists:
        for weight in weights:
            for round2 in (False, True):
                scores, improvements = [], []
                for seed in SEEDS:
                    base = base_by_seed[seed]
                    candidate = candidate_oof[name][seed]
                    blend = (1 - weight) * base + weight * candidate
                    if round2:
                        blend = np.round(blend, 2)
                    blend = np.clip(blend, 0, 1)
                    base_score = mean_absolute_error(target, base)
                    score = mean_absolute_error(target, blend)
                    scores.append(score)
                    improvements.append(base_score - score)
                rows.append({
                    "variant": name,
                    "physiology_weight": weight,
                    "baseline_weight": 1 - weight,
                    "round2": round2,
                    "mean_MAE": float(np.mean(scores)),
                    "worst_seed_MAE": float(np.max(scores)),
                    "seed_42": scores[0], "seed_77": scores[1], "seed_2026": scores[2],
                    "mean_improvement": float(np.mean(improvements)),
                    "worst_improvement": float(np.min(improvements)),
                    "wins": int(sum(x > 0 for x in improvements)),
                })

    result = pd.DataFrame(rows).sort_values(
        ["wins", "worst_improvement", "mean_MAE"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    result_path = RESULT_DIR / "team_v37_stage2_physiology_blend.csv"
    result.to_csv(result_path, index=False)
    best = result.iloc[0]
    best_mean = result[result.wins == 3].sort_values("mean_MAE").iloc[0]

    print("\n================ Team V37 최종 결과 ================")
    print(f"안정성 우선 피처군 : {best.variant}")
    print(f"기존/후보 가중치   : {best.baseline_weight:.2f} / {best.physiology_weight:.2f}")
    print(f"반올림              : {bool(best.round2)}")
    print(f"평균 CV MAE         : {best.mean_MAE:.6f}")
    print(f"최악 시드 개선      : {best.worst_improvement:+.6f}")
    print(f"승리 시드           : {int(best.wins)}/3")
    print("-----------------------------------------------------")
    print(f"평균 최저 피처군    : {best_mean.variant}")
    print(f"기존/후보 가중치   : {best_mean.baseline_weight:.2f} / {best_mean.physiology_weight:.2f}")
    print(f"반올림              : {bool(best_mean.round2)}")
    print(f"최저 평균 CV MAE    : {best_mean.mean_MAE:.6f}")
    print(f"최악 시드 개선      : {best_mean.worst_improvement:+.6f}")
    print("=====================================================")
    print(f"Screen: {Path(screen_path).resolve()}")
    print(f"Blend : {Path(result_path).resolve()}")


if __name__ == "__main__":
    main()
