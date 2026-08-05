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


def add_work_features(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    x = frame.copy()
    if variant == "baseline":
        return x

    work = pd.to_numeric(x["mean_working"], errors="coerce")
    observed = work.notna().astype(float)
    # 모든 값은 해당 행 안에서만 계산한다. 결측 대체 자체는 기존 fold pipeline이 담당한다.
    x["work_observed"] = observed
    x["work_over_8"] = np.maximum(work - 8.0, 0.0)
    x["work_over_10"] = np.maximum(work - 10.0, 0.0)
    x["work_over_12"] = np.maximum(work - 12.0, 0.0)
    x["long_work_10"] = (work > 10.0).astype(float)
    x["long_work_12"] = (work > 12.0).astype(float)
    x["short_work_7"] = ((work <= 7.0) & work.notna()).astype(float)

    if variant in {"work_sleep", "all"}:
        sleep = x["sleep_pattern"].fillna("missing").astype(str)
        difficult = (sleep == "sleep difficulty").astype(float)
        oversleep = (sleep == "oversleeping").astype(float)
        x["long_work_x_sleep_difficulty"] = x["long_work_10"] * difficult
        x["work_over_8_x_sleep_difficulty"] = x["work_over_8"] * difficult
        x["long_work_x_oversleeping"] = x["long_work_10"] * oversleep
        x["work_sleep_group"] = (
            pd.cut(
                work,
                bins=[-np.inf, 7.0, 8.0, 9.0, 10.0, np.inf],
                labels=["le7", "8", "9", "10", "gt10"],
            )
            .astype("object")
            .fillna("missing")
            .astype(str)
            + "__"
            + sleep
        )

    if variant in {"work_sex", "all"}:
        female = (x["gender"] == "F").astype(float)
        male = (x["gender"] == "M").astype(float)
        x["female_x_work_over_8"] = female * x["work_over_8"]
        x["male_x_work_over_8"] = male * x["work_over_8"]
        x["female_x_long_work"] = female * x["long_work_10"]
        x["male_x_long_work"] = male * x["long_work_10"]

    if variant in {"work_health", "all"}:
        x["work_over_8_x_allostatic"] = x["work_over_8"] * x["allostatic_count"]
        x["long_work_x_high_bp"] = x["long_work_10"] * x["risk_bp_140_90"]
        x["long_work_x_high_glucose"] = x["long_work_10"] * x["risk_glucose_99"]
        x["long_work_x_bmi"] = x["long_work_10"] * x["bmi"]

    return x


def cv_score(features, target, seed, c, gamma):
    oof = np.zeros(len(features), dtype=float)
    folds = []
    splitter = KFold(5, shuffle=True, random_state=seed)
    for train_idx, valid_idx in splitter.split(features):
        model = make_model(features, c, gamma)
        model.fit(features.iloc[train_idx], target[train_idx])
        pred = np.clip(model.predict(features.iloc[valid_idx]), 0.0, 1.0)
        oof[valid_idx] = pred
        folds.append(mean_absolute_error(target[valid_idx], pred))
    return mean_absolute_error(target, oof), folds


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    base = build_v10_winner(train.drop(columns=[ID_COLUMN, TARGET]))
    target = train[TARGET].to_numpy(float)

    variants = ("baseline", "thresholds", "work_sleep", "work_sex", "work_health", "all")
    configs = [
        (variant, c, gamma)
        for variant in variants
        for c in (3.0, 5.0)
        for gamma in (0.06, 0.08, 0.10)
    ]
    rows = []
    print(f"Stage 1: 근무·회복 피처 후보 {len(configs)}개를 seed=42에서 탐색합니다.")
    for index, (variant, c, gamma) in enumerate(configs, 1):
        features = add_work_features(base, variant)
        score, folds = cv_score(features, target, 42, c, gamma)
        rows.append(
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

    screen = pd.DataFrame(rows).sort_values(["screen_MAE", "fold_std"])
    screen_path = RESULT_DIR / "svr_v12_stage1_work_recovery_screen.csv"
    screen.to_csv(screen_path, index=False)
    finalists = screen.groupby("variant", group_keys=False).head(1).sort_values("screen_MAE")

    confirmed = []
    print("\nStage 2: 각 피처군의 최고 후보를 3개 시드에서 V11과 비교합니다.")
    for index, row in enumerate(finalists.itertuples(index=False), 1):
        features = add_work_features(base, row.variant)
        scores, fold_scores, improvements = [], [], []
        for seed in SEEDS:
            score, folds = cv_score(features, target, seed, float(row.C), float(row.gamma))
            scores.append(score)
            fold_scores.extend(folds)
            improvements.append(V11_BY_SEED[seed] - score)
        confirmed.append(
            {
                "variant": row.variant,
                "C": row.C,
                "gamma": row.gamma,
                "mean_MAE": float(np.mean(scores)),
                "worst_seed_MAE": float(np.max(scores)),
                "fold_std": float(np.std(fold_scores)),
                "seed_42": scores[0],
                "seed_77": scores[1],
                "seed_2026": scores[2],
                "mean_improvement_vs_v11": float(np.mean(improvements)),
                "worst_improvement_vs_v11": float(np.min(improvements)),
                "wins_vs_v11": int(sum(value > 0 for value in improvements)),
            }
        )
        print(
            f"[Confirm {index}/{len(finalists)}] {row.variant:<12} "
            f"평균={np.mean(scores):.6f} | 최악 개선={np.min(improvements):+.6f} | "
            f"V11 승리={sum(value > 0 for value in improvements)}/3",
            flush=True,
        )

    result = pd.DataFrame(confirmed).sort_values(
        ["mean_MAE", "worst_improvement_vs_v11", "fold_std"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    result_path = RESULT_DIR / "svr_v12_stage2_work_recovery_confirm.csv"
    result.to_csv(result_path, index=False)
    best = result.iloc[0]

    print("\n================ SVR V12 최종 결과 ================")
    print(f"최고 피처군     : {best['variant']}")
    print(f"최고 설정       : C={best['C']}, gamma={best['gamma']}, epsilon=0")
    print(f"평균 CV MAE     : {best['mean_MAE']:.6f}")
    print(f"최악 시드 MAE   : {best['worst_seed_MAE']:.6f}")
    print(f"V11 대비 개선   : {best['mean_improvement_vs_v11']:+.6f}")
    print(f"최악 시드 개선  : {best['worst_improvement_vs_v11']:+.6f}")
    print(f"V11 승리 시드   : {int(best['wins_vs_v11'])}/3")
    print("====================================================")
    print(f"Screen results : {screen_path.resolve()}")
    print(f"Confirm results: {result_path.resolve()}")


if __name__ == "__main__":
    main()
