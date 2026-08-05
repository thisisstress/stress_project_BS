from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from experiment_svr_literature_features import add_literature_features


DATA_DIR = Path("/Users/bitsaem/Desktop/stress_project/open")
RESULT_DIR = Path("/Users/bitsaem/Desktop/stress_project/8/5/8/5_result")
TARGET = "stress_score"
ID_COLUMN = "ID"
SEEDS = (42, 77, 2026)
V5_BY_SEED = {42: 0.1700149071726121, 77: 0.1677508872087785, 2026: 0.1690932446336339}


LITERATURE_AGE_COPIES = {
    "bmi": 1,
    "mean_arterial_pressure": 1,
    "allostatic_count": 2,
    "age_allostatic": 2,
    "age_bmi": 1,
    "age_map": 1,
    "age_glucose": 1,
    "age_bone_density": 1,
}


def add_weight_copies(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column, count in LITERATURE_AGE_COPIES.items():
        for index in range(1, count + 1):
            result[f"{column}__weight_copy_{index}"] = result[column]
    return result


def add_interactions(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    x = frame.copy()
    female = (x["gender"] == "F").astype(float)
    male = (x["gender"] == "M").astype(float)

    if variant in {"sex", "all"}:
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

    if variant in {"age_band", "all"}:
        x["age_squared"] = x["age"].pow(2)
        x["age_band"] = pd.cut(
            x["age"],
            bins=[-np.inf, 29, 44, 59, 74, np.inf],
            labels=["17_29", "30_44", "45_59", "60_74", "75_plus"],
        ).astype(str)
        x["gender_age_band"] = x["gender"].astype(str) + "__" + x["age_band"]
        x["age_risk_count"] = x["age"] * x["allostatic_count"]

    if variant in {"history", "all"}:
        personal = x["medical_history"].fillna("none").astype(str)
        family = x["family_medical_history"].fillna("none").astype(str)
        x["personal_family_pair"] = personal + "__" + family
        x["same_history"] = ((personal == family) & (personal != "none")).astype(float)
        x["history_count"] = (personal != "none").astype(float) + (
            family != "none"
        ).astype(float)
        x["age_history_count"] = x["age"] * x["history_count"]
        x["allostatic_history"] = x["allostatic_count"] * x["history_count"]

    return add_weight_copies(x)


def make_model(frame: pd.DataFrame, c: float, gamma: float) -> Pipeline:
    categorical = frame.select_dtypes(exclude="number").columns.tolist()
    numerical = frame.select_dtypes(include="number").columns.tolist()
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", SVR(C=c, gamma=gamma, epsilon=0.0, kernel="rbf", cache_size=1500)),
        ]
    )


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
    base = add_literature_features(
        train.drop(columns=[ID_COLUMN, TARGET]), "physiology"
    )
    target = train[TARGET].to_numpy(float)

    variants = ("baseline", "sex", "age_band", "history", "all")
    configs = [
        (variant, c, gamma)
        for variant in variants
        for c in (1.0, 2.0)
        for gamma in (0.12, 0.14, 0.16)
    ]
    screen_rows = []
    print(f"Stage 1: 인구학적 상호작용 후보 {len(configs)}개를 탐색합니다.")
    for index, (variant, c, gamma) in enumerate(configs, 1):
        features = add_interactions(base, "none" if variant == "baseline" else variant)
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
            f"[{index:02d}/{len(configs):02d}] {variant:<9} C={c} gamma={gamma} "
            f"MAE={score:.6f}",
            flush=True,
        )

    screen = pd.DataFrame(screen_rows).sort_values(["screen_MAE", "fold_std"])
    screen.to_csv(RESULT_DIR / "svr_v6_stage1_interaction_screen.csv", index=False)
    finalists = screen.groupby("variant", group_keys=False).head(1).sort_values("screen_MAE")

    confirm_rows = []
    print("\nStage 2: 각 피처군 최고 후보를 3개 시드로 확인합니다.")
    for index, row in enumerate(finalists.itertuples(index=False), 1):
        features = add_interactions(base, "none" if row.variant == "baseline" else row.variant)
        seed_scores = []
        fold_scores = []
        improvements = []
        for seed in SEEDS:
            score, folds = cv_score(features, target, seed, float(row.C), float(row.gamma))
            seed_scores.append(score)
            fold_scores.extend(folds)
            improvements.append(V5_BY_SEED[seed] - score)
        confirm_rows.append(
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
                "mean_improvement_vs_v5": float(np.mean(improvements)),
                "worst_improvement_vs_v5": float(np.min(improvements)),
                "wins_vs_v5": int(sum(value > 0 for value in improvements)),
            }
        )
        print(
            f"[Confirm {index}/{len(finalists)}] {row.variant:<9} "
            f"평균={np.mean(seed_scores):.6f} | 개선={np.mean(improvements):+.6f} | "
            f"V5 승리={sum(value > 0 for value in improvements)}/3",
            flush=True,
        )

    result = pd.DataFrame(confirm_rows).sort_values(
        ["mean_MAE", "worst_improvement_vs_v5", "fold_std"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    path = RESULT_DIR / "svr_v6_stage2_interaction_confirm.csv"
    result.to_csv(path, index=False)
    best = result.iloc[0]
    print("\n================ SVR V6 최종 결과 ================")
    print(f"최고 피처군   : {best['variant']}")
    print(f"최고 설정     : C={best['C']}, gamma={best['gamma']}, epsilon=0")
    print(f"평균 CV MAE   : {best['mean_MAE']:.6f}")
    print(f"최악 시드 MAE : {best['worst_seed_MAE']:.6f}")
    print(f"V5 대비 개선  : {best['mean_improvement_vs_v5']:+.6f}")
    print(f"최악 개선량   : {best['worst_improvement_vs_v5']:+.6f}")
    print(f"V5 승리 시드  : {int(best['wins_vs_v5'])}/3")
    print("===================================================")
    print(f"Saved: {path.resolve()}")


if __name__ == "__main__":
    main()
