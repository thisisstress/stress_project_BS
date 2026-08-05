from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from experiment_svr_demographic_interactions_v6 import add_interactions, make_model
from experiment_svr_literature_features import add_literature_features


DATA_DIR = Path("/Users/bitsaem/Desktop/stress_project/open")
RESULT_DIR = Path("/Users/bitsaem/Desktop/stress_project/8/5/8/5_result")
TARGET = "stress_score"
ID_COLUMN = "ID"
SEEDS = (42, 77, 2026)
V6_BY_SEED = {42: 0.169519, 77: 0.167309, 2026: 0.168028}


def add_ordinal_features(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    x = frame.copy()

    if variant in {"activity", "all"}:
        x["activity_intensity"] = x["activity"].map(
            {"light": 0.0, "moderate": 1.0, "intense": 2.0}
        )
        x["activity_distance_moderate"] = (x["activity_intensity"] - 1.0).abs()

    if variant in {"smoking", "all"}:
        x["smoking_exposure"] = x["smoke_status"].map(
            {"non-smoker": 0.0, "ex-smoker": 1.0, "current-smoker": 2.0}
        )
        x["current_smoker_flag"] = (x["smoke_status"] == "current-smoker").astype(float)

    if variant in {"sleep", "all"}:
        # 정상에서 벗어났는지와 유형을 분리한다. 강제적인 완전 순서는 사용하지 않는다.
        x["sleep_disruption"] = (x["sleep_pattern"] != "normal").astype(float)
        x["sleep_difficulty_flag"] = (
            x["sleep_pattern"] == "sleep difficulty"
        ).astype(float)
        x["oversleeping_flag"] = (x["sleep_pattern"] == "oversleeping").astype(float)

    if variant in {"education", "all"}:
        x["education_level_numeric"] = x["edu_level"].map(
            {
                "high school diploma": 0.0,
                "bachelors degree": 1.0,
                "graduate degree": 2.0,
            }
        )
        x["education_missing"] = x["edu_level"].isna().astype(float)

    if variant in {"work_missing", "all"}:
        x["working_missing"] = x["mean_working"].isna().astype(float)
        x["working_distance_8"] = (x["mean_working"] - 8.0).abs()
        x["working_over_8"] = np.maximum(x["mean_working"] - 8.0, 0.0)

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
    # V6 우승 구성: history 상호작용과 literature_age 거리 가중치.
    v6_base = add_interactions(physiology, "history")
    target = train[TARGET].to_numpy(float)

    variants = (
        "baseline",
        "activity",
        "smoking",
        "sleep",
        "education",
        "work_missing",
        "all",
    )
    configs = [
        (variant, c, gamma)
        for variant in variants
        for c in (1.0, 2.0)
        for gamma in (0.12, 0.14, 0.16)
    ]
    screen_rows = []
    print(f"Stage 1: ordinal/domain 후보 {len(configs)}개를 탐색합니다.")
    for index, (variant, c, gamma) in enumerate(configs, 1):
        features = add_ordinal_features(v6_base, "none" if variant == "baseline" else variant)
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
            f"[{index:02d}/{len(configs):02d}] {variant:<12} C={c} gamma={gamma} "
            f"MAE={score:.6f}",
            flush=True,
        )

    screen = pd.DataFrame(screen_rows).sort_values(["screen_MAE", "fold_std"])
    screen.to_csv(RESULT_DIR / "svr_v8_stage1_ordinal_screen.csv", index=False)
    finalists = screen.groupby("variant", group_keys=False).head(1).sort_values("screen_MAE")

    rows = []
    print("\nStage 2: 각 ordinal 피처군 최고 후보를 3개 시드로 확인합니다.")
    for index, row in enumerate(finalists.itertuples(index=False), 1):
        features = add_ordinal_features(v6_base, "none" if row.variant == "baseline" else row.variant)
        seed_scores = []
        fold_scores = []
        improvements = []
        for seed in SEEDS:
            score, folds = cv_score(features, target, seed, float(row.C), float(row.gamma))
            seed_scores.append(score)
            fold_scores.extend(folds)
            improvements.append(V6_BY_SEED[seed] - score)
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
                "mean_improvement_vs_v6": float(np.mean(improvements)),
                "worst_improvement_vs_v6": float(np.min(improvements)),
                "wins_vs_v6": int(sum(value > 0 for value in improvements)),
            }
        )
        print(
            f"[Confirm {index}/{len(finalists)}] {row.variant:<12} "
            f"평균={np.mean(seed_scores):.6f} | 개선={np.mean(improvements):+.6f} | "
            f"V6 승리={sum(value > 0 for value in improvements)}/3",
            flush=True,
        )

    result = pd.DataFrame(rows).sort_values(
        ["mean_MAE", "worst_improvement_vs_v6", "fold_std"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    path = RESULT_DIR / "svr_v8_stage2_ordinal_confirm.csv"
    result.to_csv(path, index=False)
    best = result.iloc[0]

    print("\n================ SVR V8 최종 결과 ================")
    print(f"최고 피처군   : {best['variant']}")
    print(f"최고 설정     : C={best['C']}, gamma={best['gamma']}, epsilon=0")
    print(f"평균 CV MAE   : {best['mean_MAE']:.6f}")
    print(f"최악 시드 MAE : {best['worst_seed_MAE']:.6f}")
    print(f"V6 대비 개선  : {best['mean_improvement_vs_v6']:+.6f}")
    print(f"최악 개선량   : {best['worst_improvement_vs_v6']:+.6f}")
    print(f"V6 승리 시드  : {int(best['wins_vs_v6'])}/3")
    print("===================================================")
    print(f"Saved: {path.resolve()}")


if __name__ == "__main__":
    main()
