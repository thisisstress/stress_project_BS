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
    build_variant,
    load_team_functions,
)


def paired_oof(raw, target, seed, add_features, tree_predict, pair_predict, pair_weight):
    base_x = build_variant(raw, "baseline", add_features)
    physiology_x = build_variant(raw, "physiology", add_features)
    base_oof = np.zeros(len(raw), dtype=float)
    physiology_oof = np.zeros(len(raw), dtype=float)
    splitter = KFold(5, shuffle=True, random_state=seed)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(raw), 1):
        train_y = target.iloc[train_idx]
        predictions = []
        for features in (base_x, physiology_x):
            tree = tree_predict(
                features.iloc[train_idx], features.iloc[valid_idx], train_y
            )
            pair = pair_predict(
                features.iloc[train_idx], features.iloc[valid_idx], train_y
            )
            predictions.append(np.clip(np.round(
                (1.0 - pair_weight) * tree + pair_weight * pair, 2
            ), 0.0, 1.0))
        base_oof[valid_idx], physiology_oof[valid_idx] = predictions
        print(f"Seed {seed} | Fold {fold}/5 complete", flush=True)
    return base_oof, physiology_oof


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    add_features, tree_predict, pair_predict, pair_weight = load_team_functions()
    train = pd.read_csv(DATA_DIR / "train.csv")
    raw = train.drop(columns=[ID_COLUMN, TARGET])
    target = train[TARGET].reset_index(drop=True)

    predictions = {}
    for seed in SEEDS:
        predictions[seed] = paired_oof(
            raw, target, seed, add_features, tree_predict, pair_predict, pair_weight
        )

    rows = []
    for physiology_weight in np.round(np.arange(0.0, 0.55, 0.05), 2):
        for round2 in (False, True):
            scores, improvements = [], []
            for seed in SEEDS:
                base, physiology = predictions[seed]
                blend = (
                    (1.0 - physiology_weight) * base
                    + physiology_weight * physiology
                )
                if round2:
                    blend = np.round(blend, 2)
                blend = np.clip(blend, 0.0, 1.0)
                base_score = mean_absolute_error(target, base)
                score = mean_absolute_error(target, blend)
                scores.append(score)
                improvements.append(base_score - score)
            rows.append({
                "physiology_weight": physiology_weight,
                "baseline_weight": 1.0 - physiology_weight,
                "round2": round2,
                "mean_MAE": float(np.mean(scores)),
                "worst_seed_MAE": float(np.max(scores)),
                "seed_42": scores[0],
                "seed_77": scores[1],
                "seed_2026": scores[2],
                "mean_improvement": float(np.mean(improvements)),
                "worst_improvement": float(np.min(improvements)),
                "wins": int(sum(value > 0 for value in improvements)),
            })
            print(
                f"physiology={physiology_weight:.2f} round2={str(round2):<5} | "
                f"평균={np.mean(scores):.6f} | 최악 개선={np.min(improvements):+.6f} | "
                f"승리={sum(value > 0 for value in improvements)}/3"
            )

    result = pd.DataFrame(rows).sort_values(
        ["wins", "worst_improvement", "mean_MAE"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    path = RESULT_DIR / "team_v36_physiology_blend.csv"
    result.to_csv(path, index=False)
    best = result.iloc[0]

    print("\n================ Team V36 최종 결과 ================")
    print(f"기존/생리 가중치 : {best.baseline_weight:.2f} / {best.physiology_weight:.2f}")
    print(f"소수 둘째자리 반올림: {bool(best.round2)}")
    print(f"평균 CV MAE      : {best.mean_MAE:.6f}")
    print(f"최악 시드 MAE    : {best.worst_seed_MAE:.6f}")
    print(f"평균 개선        : {best.mean_improvement:+.6f}")
    print(f"최악 시드 개선   : {best.worst_improvement:+.6f}")
    print(f"기준 승리 시드   : {int(best.wins)}/3")
    print("=====================================================")
    print(f"Saved: {Path(path).resolve()}")


if __name__ == "__main__":
    main()
