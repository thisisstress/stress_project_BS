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


def add_domain_indices(raw: pd.DataFrame) -> pd.DataFrame:
    x = raw.copy()
    height_m = x["height"] / 100.0
    bmi = x["weight"] / height_m.pow(2)
    sbp = x["systolic_blood_pressure"]
    dbp = x["diastolic_blood_pressure"]

    # 임상 경계로부터 얼마나 벗어났는지를 연속형 부담으로 표현한다.
    bmi_load = np.maximum(bmi - 25.0, 0.0) / 5.0
    sbp_load = np.maximum(sbp - 120.0, 0.0) / 20.0
    dbp_load = np.maximum(dbp - 80.0, 0.0) / 10.0
    glucose_load = np.maximum(x["glucose"] - 99.0, 0.0) / 30.0
    cholesterol_load = np.maximum(x["cholesterol"] - 200.0, 0.0) / 50.0
    bone_load = np.maximum(0.0 - x["bone_density"], 0.0)

    activity = x["activity"].map({"intense": 0.0, "moderate": 0.5, "light": 1.0})
    smoking = x["smoke_status"].map(
        {"non-smoker": 0.0, "ex-smoker": 0.5, "current-smoker": 1.0}
    )
    sleep = x["sleep_pattern"].map(
        {"normal": 0.0, "oversleeping": 0.5, "sleep difficulty": 1.0}
    )
    work = pd.to_numeric(x["mean_working"], errors="coerce")
    work_load = np.maximum(work - 8.0, 0.0) / 4.0

    personal_history = x["medical_history"].notna().astype(float)
    family_history = x["family_medical_history"].notna().astype(float)
    age_load = np.maximum(x["age"] - 40.0, 0.0) / 30.0

    x["cardiometabolic_load_index"] = (
        bmi_load + sbp_load + dbp_load + glucose_load + cholesterol_load
    ) / 5.0
    x["physiological_load_index"] = (
        bmi_load
        + sbp_load
        + dbp_load
        + glucose_load
        + cholesterol_load
        + bone_load
        + age_load
    ) / 7.0
    x["lifestyle_load_index"] = (
        activity + smoking + sleep + work_load
    ) / 4.0
    x["history_load_index"] = (
        personal_history + 0.75 * family_history
    ) / 1.75
    x["recovery_deficit_index"] = (
        sleep + activity + work_load
    ) / 3.0
    x["domain_stress_burden"] = (
        0.35 * x["physiological_load_index"]
        + 0.40 * x["lifestyle_load_index"]
        + 0.25 * x["history_load_index"]
    )
    x["load_x_recovery_deficit"] = (
        x["physiological_load_index"] * x["recovery_deficit_index"]
    )
    x["burden_x_age"] = x["domain_stress_burden"] * age_load
    x["burden_x_female"] = (
        x["domain_stress_burden"] * (x["gender"] == "F").astype(float)
    )
    return x


def build_variant(raw: pd.DataFrame, variant: str) -> pd.DataFrame:
    indexed = add_domain_indices(raw)
    index_columns = [
        "cardiometabolic_load_index",
        "physiological_load_index",
        "lifestyle_load_index",
        "history_load_index",
        "recovery_deficit_index",
        "domain_stress_burden",
        "load_x_recovery_deficit",
        "burden_x_age",
        "burden_x_female",
    ]
    if variant == "raw_plus_indices":
        return indexed
    if variant == "indices_weighted":
        result = indexed.copy()
        for column in index_columns:
            result[f"{column}__copy"] = result[column]
        return result
    if variant == "indices_only":
        keep = ["gender", "age"] + index_columns
        return indexed[keep].copy()
    if variant == "v11_full":
        return build_v10_winner(raw)
    if variant in {"v11_plus_indices", "v11_plus_weighted_indices"}:
        result = build_v10_winner(raw)
        for column in index_columns:
            result[column] = indexed[column]
            if variant == "v11_plus_weighted_indices":
                result[f"{column}__copy"] = indexed[column]
        return result
    raise ValueError(variant)


def cv_score(features, target, seed, c, gamma):
    oof = np.zeros(len(features), dtype=float)
    folds = []
    for train_idx, valid_idx in KFold(5, shuffle=True, random_state=seed).split(features):
        model = make_model(features, c, gamma)
        model.fit(features.iloc[train_idx], target[train_idx])
        pred = np.clip(model.predict(features.iloc[valid_idx]), 0.0, 1.0)
        oof[valid_idx] = pred
        folds.append(mean_absolute_error(target[valid_idx], pred))
    return mean_absolute_error(target, oof), folds


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    raw = train.drop(columns=[ID_COLUMN, TARGET])
    target = train[TARGET].to_numpy(float)

    variants = (
        "raw_plus_indices",
        "indices_weighted",
        "indices_only",
        "v11_full",
        "v11_plus_indices",
        "v11_plus_weighted_indices",
    )
    configs = [
        (variant, c, gamma)
        for variant in variants
        for c in (0.5, 1.0, 2.0, 5.0)
        for gamma in (0.03, 0.06, 0.09)
    ]
    screen_rows = []
    print(f"Stage 1: 도메인 합성지수 SVR 후보 {len(configs)}개를 seed=42에서 탐색합니다.")
    for index, (variant, c, gamma) in enumerate(configs, 1):
        features = build_variant(raw, variant)
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
            f"[{index:02d}/{len(configs):02d}] {variant:<27} "
            f"features={features.shape[1]:<2} C={c} gamma={gamma} MAE={score:.6f}",
            flush=True,
        )

    screen = pd.DataFrame(screen_rows).sort_values(["screen_MAE", "fold_std"])
    screen_path = RESULT_DIR / "svr_v16_stage1_domain_composite.csv"
    screen.to_csv(screen_path, index=False)
    finalists = screen.groupby("variant", group_keys=False).head(1).sort_values("screen_MAE")

    rows = []
    print("\nStage 2: 구성별 최고 후보를 3개 시드에서 V11과 비교합니다.")
    for index, row in enumerate(finalists.itertuples(index=False), 1):
        features = build_variant(raw, row.variant)
        scores, all_folds, improvements = [], [], []
        for seed in SEEDS:
            score, folds = cv_score(features, target, seed, float(row.C), float(row.gamma))
            scores.append(score)
            all_folds.extend(folds)
            improvements.append(V11_BY_SEED[seed] - score)
        rows.append(
            {
                "variant": row.variant,
                "C": row.C,
                "gamma": row.gamma,
                "n_features": row.n_features,
                "mean_MAE": float(np.mean(scores)),
                "worst_seed_MAE": float(np.max(scores)),
                "fold_std": float(np.std(all_folds)),
                "seed_42": scores[0],
                "seed_77": scores[1],
                "seed_2026": scores[2],
                "mean_improvement_vs_v11": float(np.mean(improvements)),
                "worst_improvement_vs_v11": float(np.min(improvements)),
                "wins_vs_v11": int(sum(value > 0 for value in improvements)),
            }
        )
        print(
            f"[Confirm {index}/{len(finalists)}] {row.variant:<27} "
            f"평균={np.mean(scores):.6f} | 최악 개선={np.min(improvements):+.6f} | "
            f"V11 승리={sum(value > 0 for value in improvements)}/3",
            flush=True,
        )

    result = pd.DataFrame(rows).sort_values(
        ["mean_MAE", "worst_improvement_vs_v11", "fold_std"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    result_path = RESULT_DIR / "svr_v16_stage2_domain_composite_confirm.csv"
    result.to_csv(result_path, index=False)
    best = result.iloc[0]
    competitive = bool(best["mean_MAE"] <= 0.145)

    print("\n================ SVR V16 최종 결과 ================")
    print(f"최고 피처 구성   : {best['variant']}")
    print(f"피처 수          : {int(best['n_features'])}")
    print(f"최고 설정        : C={best['C']}, gamma={best['gamma']}, epsilon=0")
    print(f"평균 CV MAE      : {best['mean_MAE']:.6f}")
    print(f"최악 시드 MAE    : {best['worst_seed_MAE']:.6f}")
    print(f"V11 대비 개선    : {best['mean_improvement_vs_v11']:+.6f}")
    print(f"V11 승리 시드    : {int(best['wins_vs_v11'])}/3")
    print(f"경쟁 기준 0.145 통과: {competitive}")
    print("====================================================")
    print(f"Screen results : {screen_path.resolve()}")
    print(f"Confirm results: {result_path.resolve()}")


if __name__ == "__main__":
    main()
