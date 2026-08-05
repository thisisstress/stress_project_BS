from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.svm import SVR


DATA_DIR = Path("/Users/bitsaem/Desktop/stress_project/open")
RESULT_DIR = Path("/Users/bitsaem/Desktop/stress_project/8/5/8/5_result")
TARGET = "stress_score"
ID_COLUMN = "ID"
SEEDS = (42, 77, 2026)


def add_literature_features(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    """논문에서 제시된 관계를 현재 행의 기존 변수만으로 표현한다."""
    x = frame.copy()
    height_m = x["height"] / 100.0
    x["bmi"] = x["weight"] / height_m.pow(2)

    if group in {"physiology", "full"}:
        sbp = x["systolic_blood_pressure"]
        dbp = x["diastolic_blood_pressure"]
        x["pulse_pressure"] = sbp - dbp
        x["mean_arterial_pressure"] = (sbp + 2.0 * dbp) / 3.0
        x["bp_ratio"] = sbp / (dbp + 1e-6)
        x["cholesterol_glucose_ratio"] = x["cholesterol"] / (
            x["glucose"] + 1e-6
        )
        x["metabolic_product"] = (
            x["bmi"] * x["glucose"] * x["cholesterol"]
        ) ** (1.0 / 3.0)

        # Allostatic-load 연구의 임상 위험 경계. 외부 관측값은 사용하지 않는다.
        x["risk_bmi_25"] = (x["bmi"] >= 25.0).astype(float)
        x["risk_bmi_30"] = (x["bmi"] >= 30.0).astype(float)
        x["risk_bp_140_90"] = ((sbp >= 140) | (dbp >= 90)).astype(float)
        x["risk_glucose_99"] = (x["glucose"] >= 99.0).astype(float)
        x["risk_cholesterol_250"] = (x["cholesterol"] >= 250.0).astype(float)
        x["allostatic_count"] = x[
            [
                "risk_bmi_25",
                "risk_bp_140_90",
                "risk_glucose_99",
                "risk_cholesterol_250",
            ]
        ].sum(axis=1)
        x["age_allostatic"] = x["age"] * x["allostatic_count"]
        x["age_bmi"] = x["age"] * x["bmi"]
        x["age_map"] = x["age"] * x["mean_arterial_pressure"]
        x["age_glucose"] = x["age"] * x["glucose"]
        x["age_bone_density"] = x["age"] * x["bone_density"]

    if group in {"recovery", "full"}:
        working = x["mean_working"]
        sleep_bad = x["sleep_pattern"].isin(
            ["sleep difficulty", "oversleeping"]
        )
        x["work_over_8"] = np.maximum(working - 8.0, 0.0)
        x["work_under_8"] = np.maximum(8.0 - working, 0.0)
        x["work_distance_8"] = np.abs(working - 8.0)
        x["work_squared"] = working.pow(2)
        x["sleep_disrupted"] = sleep_bad.astype(float)
        x["sleep_difficulty"] = (
            x["sleep_pattern"] == "sleep difficulty"
        ).astype(float)
        x["oversleeping"] = (x["sleep_pattern"] == "oversleeping").astype(float)
        x["work_sleep_burden"] = x["work_over_8"] * x["sleep_disrupted"]
        x["work_sleep_group"] = (
            x["sleep_pattern"].fillna("missing").astype(str)
            + "__"
            + pd.cut(
                working,
                bins=[-np.inf, 7.0, 9.0, np.inf],
                labels=["short", "standard", "long"],
            ).astype(str)
        )

    if group == "full":
        x["current_smoker"] = (x["smoke_status"] == "current-smoker").astype(float)
        x["low_activity"] = (x["activity"] == "light").astype(float)
        x["intense_activity"] = (x["activity"] == "intense").astype(float)
        x["personal_history"] = x["medical_history"].notna().astype(float)
        x["family_history"] = x["family_medical_history"].notna().astype(float)
        x["history_burden"] = x["personal_history"] + x["family_history"]
        x["lifestyle_burden"] = (
            x["current_smoker"]
            + x["low_activity"]
            + x["sleep_disrupted"]
            + (x["work_over_8"] > 0).astype(float)
        )
        x["total_burden"] = (
            x["allostatic_count"] + x["history_burden"] + x["lifestyle_burden"]
        )

    x["missing_count"] = x.isna().sum(axis=1)
    return x


def make_model(frame: pd.DataFrame, c: float, gamma: str | float) -> Pipeline:
    categorical = frame.select_dtypes(exclude="number").columns.tolist()
    numerical = frame.select_dtypes(include="number").columns.tolist()
    preprocessor = ColumnTransformer(
        [
            (
                "numerical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", RobustScaler()),
                    ]
                ),
                numerical,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="missing"),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                SVR(kernel="rbf", C=c, gamma=gamma, epsilon=0.01, cache_size=1000),
            ),
        ]
    )


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    raw = train.drop(columns=[ID_COLUMN, TARGET])
    target = train[TARGET].to_numpy(float)

    # 작은 사전 고정 grid: 피처군 효과 확인이 목적이며 대규모 CV 튜닝을 피한다.
    configs = [(1.0, "scale"), (3.0, "scale"), (10.0, "scale"), (3.0, 0.03)]
    rows: list[dict[str, object]] = []
    for group in ("base", "physiology", "recovery", "full"):
        features = add_literature_features(raw, group)
        for c, gamma in configs:
            seed_scores = []
            all_fold_scores = []
            for seed in SEEDS:
                oof = np.zeros(len(features), dtype=float)
                cv = KFold(n_splits=5, shuffle=True, random_state=seed)
                for train_idx, valid_idx in cv.split(features):
                    model = make_model(features, c, gamma)
                    model.fit(features.iloc[train_idx], target[train_idx])
                    oof[valid_idx] = np.clip(
                        model.predict(features.iloc[valid_idx]), 0.0, 1.0
                    )
                    all_fold_scores.append(
                        mean_absolute_error(target[valid_idx], oof[valid_idx])
                    )
                seed_scores.append(mean_absolute_error(target, oof))
            row = {
                "feature_group": group,
                "C": c,
                "gamma": gamma,
                "mean_MAE": float(np.mean(seed_scores)),
                "worst_seed_MAE": float(np.max(seed_scores)),
                "fold_std": float(np.std(all_fold_scores)),
                "seed_42": seed_scores[0],
                "seed_77": seed_scores[1],
                "seed_2026": seed_scores[2],
            }
            rows.append(row)
            print(
                f"[{group:10s}] C={c:<4} gamma={str(gamma):<5} | "
                f"평균 MAE={row['mean_MAE']:.6f} | "
                f"최악 시드 MAE={row['worst_seed_MAE']:.6f} | "
                f"시드별={np.round(seed_scores, 6).tolist()}",
                flush=True,
            )

    result = pd.DataFrame(rows).sort_values(
        ["mean_MAE", "worst_seed_MAE", "fold_std"]
    )
    output = RESULT_DIR / "svr_literature_feature_results.csv"
    result.to_csv(output, index=False)
    best = result.iloc[0]
    print("\n========== SVR 논문 기반 피처 실험 결과 ==========")
    print("상위 후보")
    print(result.head(12).to_string(index=False))
    print("\n---------------- 최종 최고 결과 ----------------")
    print(f"최고 피처 구성 : {best['feature_group']}")
    print(f"최고 설정      : C={best['C']}, gamma={best['gamma']}")
    print(f"평균 CV MAE    : {best['mean_MAE']:.6f}")
    print(f"최악 시드 MAE  : {best['worst_seed_MAE']:.6f}")
    print(
        "시드별 MAE    : "
        f"42={best['seed_42']:.6f}, "
        f"77={best['seed_77']:.6f}, "
        f"2026={best['seed_2026']:.6f}"
    )
    print("==================================================")
    print(f"\nSaved: {output.resolve()}")


if __name__ == "__main__":
    main()
