from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from experiment_team_feature_transfer_v35 import (
    DATA_DIR, ID_COLUMN, RESULT_DIR, SEEDS, TARGET, load_team_functions,
)
from experiment_svr_v10_kernel_tuning_v11 import build_v10_winner


AGE = {"age_bmi", "age_map", "age_glucose", "age_bone_density"}
METABOLIC = {"metabolic_product"}
VARIANTS = {
    "age_full4": AGE,
    "drop_age_bmi": AGE - {"age_bmi"},
    "drop_age_map": AGE - {"age_map"},
    "drop_age_glucose": AGE - {"age_glucose"},
    "drop_age_bone": AGE - {"age_bone_density"},
    "age_bmi_only": {"age_bmi"},
    "age_map_only": {"age_map"},
    "age_glucose_only": {"age_glucose"},
    "age_bone_only": {"age_bone_density"},
    "bmi_map": {"age_bmi", "age_map"},
    "bmi_glucose": {"age_bmi", "age_glucose"},
    "bmi_bone": {"age_bmi", "age_bone_density"},
    "map_glucose": {"age_map", "age_glucose"},
    "map_bone": {"age_map", "age_bone_density"},
    "glucose_bone": {"age_glucose", "age_bone_density"},
    "age_plus_metabolic": AGE | METABOLIC,
}


def make_features(raw, add_features, columns=None):
    base = add_features(raw).reset_index(drop=True)
    if not columns:
        return base
    source = build_v10_winner(raw).reset_index(drop=True)
    for column in columns:
        if column in source.columns and column not in base.columns:
            base[column] = source[column]
    return base


def oof_predict(features, y, seed, tree_predict, pair_predict, pair_weight):
    pred = np.zeros(len(features))
    cv = KFold(5, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(cv.split(features), 1):
        tree = tree_predict(features.iloc[tr], features.iloc[va], y.iloc[tr])
        pair = pair_predict(features.iloc[tr], features.iloc[va], y.iloc[tr])
        pred[va] = np.clip(np.round(
            (1 - pair_weight) * tree + pair_weight * pair, 2
        ), 0, 1)
        print(f"Seed {seed} | Fold {fold}/5 complete", flush=True)
    return pred


def score(y, pred):
    return mean_absolute_error(y, pred)


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    add_features, tree_predict, pair_predict, pair_weight = load_team_functions()
    train = pd.read_csv(DATA_DIR / "train.csv")
    raw = train.drop(columns=[ID_COLUMN, TARGET])
    y = train[TARGET].reset_index(drop=True)

    base_features = make_features(raw, add_features)
    print("Stage 1: 연령×생리 피처 16개 구성을 seed=42에서 탐색합니다.")
    base42 = oof_predict(base_features, y, 42, tree_predict, pair_predict, pair_weight)
    base42_mae = score(y, base42)
    screen_oof, rows = {}, []
    for idx, (name, columns) in enumerate(VARIANTS.items(), 1):
        features = make_features(raw, add_features, columns)
        pred = oof_predict(features, y, 42, tree_predict, pair_predict, pair_weight)
        mae = score(y, pred)
        screen_oof[name] = pred
        rows.append({"variant": name, "added_features": len(columns),
                     "seed42_MAE": mae, "improvement": base42_mae - mae})
        print(f"[{idx:02d}/{len(VARIANTS)}] {name:<20} MAE={mae:.6f} | "
              f"개선={base42_mae-mae:+.6f}", flush=True)
    screen = pd.DataFrame(rows).sort_values("seed42_MAE").reset_index(drop=True)
    screen_path = RESULT_DIR / "team_v38_stage1_agephys_selection.csv"
    screen.to_csv(screen_path, index=False)
    finalists = list(dict.fromkeys(
        screen.head(5).variant.tolist() + ["age_full4", "age_plus_metabolic"]
    ))

    print("\nStage 2: 상위 후보를 3시드에서 확인하고 모델 간 혼합합니다.")
    base = {42: base42}
    for seed in SEEDS[1:]:
        base[seed] = oof_predict(
            base_features, y, seed, tree_predict, pair_predict, pair_weight
        )
    preds = {}
    for name in finalists:
        features = make_features(raw, add_features, VARIANTS[name])
        preds[name] = {42: screen_oof[name]}
        for seed in SEEDS[1:]:
            preds[name][seed] = oof_predict(
                features, y, seed, tree_predict, pair_predict, pair_weight
            )

    candidates = []
    # 단일 후보와 기존 모델의 혼합
    for name in finalists:
        for w in np.round(np.arange(0.5, 1.01, 0.05), 2):
            candidates.append((f"base+{name}", name, None, w))
    # 서로 다른 age 신호 간 혼합. 첫 모델의 비율을 w로 둔다.
    for i, left in enumerate(finalists):
        for right in finalists[i + 1:]:
            for w in (0.25, 0.50, 0.75):
                candidates.append((f"{left}+{right}", left, right, w))

    result_rows = []
    for label, left, right, w in candidates:
        for round2 in (False, True):
            scores, gains = [], []
            for seed in SEEDS:
                if right is None:
                    raw_pred = (1 - w) * base[seed] + w * preds[left][seed]
                else:
                    raw_pred = w * preds[left][seed] + (1 - w) * preds[right][seed]
                pred = np.round(raw_pred, 2) if round2 else raw_pred
                pred = np.clip(pred, 0, 1)
                base_mae = score(y, base[seed])
                mae = score(y, pred)
                scores.append(mae)
                gains.append(base_mae - mae)
            result_rows.append({
                "blend": label, "left": left, "right": right or "baseline",
                "left_or_candidate_weight": w, "round2": round2,
                "mean_MAE": np.mean(scores), "worst_seed_MAE": np.max(scores),
                "seed_42": scores[0], "seed_77": scores[1], "seed_2026": scores[2],
                "mean_improvement": np.mean(gains),
                "worst_improvement": np.min(gains),
                "wins": sum(g > 0 for g in gains),
            })

    result = pd.DataFrame(result_rows).sort_values(
        ["wins", "worst_improvement", "mean_MAE"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    result_path = RESULT_DIR / "team_v38_stage2_agephys_blend.csv"
    result.to_csv(result_path, index=False)
    stable = result.iloc[0]
    aggressive = result[result.wins == 3].sort_values("mean_MAE").iloc[0]

    print("\n================ Team V38 최종 결과 ================")
    print(f"안정성 우선       : {stable.blend}")
    print(f"가중치/반올림     : {stable.left_or_candidate_weight:.2f} / {bool(stable.round2)}")
    print(f"평균 CV MAE       : {stable.mean_MAE:.6f}")
    print(f"최악 시드 개선    : {stable.worst_improvement:+.6f}")
    print(f"승리 시드         : {int(stable.wins)}/3")
    print("-----------------------------------------------------")
    print(f"평균 최저         : {aggressive.blend}")
    print(f"가중치/반올림     : {aggressive.left_or_candidate_weight:.2f} / {bool(aggressive.round2)}")
    print(f"최저 평균 CV MAE  : {aggressive.mean_MAE:.6f}")
    print(f"최악 시드 개선    : {aggressive.worst_improvement:+.6f}")
    print("=====================================================")
    print(f"Screen: {Path(screen_path).resolve()}")
    print(f"Blend : {Path(result_path).resolve()}")


if __name__ == "__main__":
    main()
