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
SCREEN_SEED = 42
CONFIRM_SEEDS = (42, 77, 2026)


AGE_FEATURES = ["age_allostatic", "age_bmi", "age_map", "age_glucose", "age_bone_density"]
THRESHOLD_FEATURES = [
    "risk_bmi_25",
    "risk_bmi_30",
    "risk_bp_140_90",
    "risk_glucose_99",
    "risk_cholesterol_250",
    "allostatic_count",
    "age_allostatic",
]
RATIO_FEATURES = [
    "pulse_pressure",
    "mean_arterial_pressure",
    "bp_ratio",
    "cholesterol_glucose_ratio",
    "metabolic_product",
]


def select_features(full: pd.DataFrame, variant: str) -> pd.DataFrame:
    if variant == "all":
        return full.copy()
    if variant == "no_age_interactions":
        return full.drop(columns=AGE_FEATURES)
    if variant == "no_thresholds":
        return full.drop(columns=THRESHOLD_FEATURES)
    if variant == "no_ratios":
        return full.drop(columns=RATIO_FEATURES)
    if variant == "core":
        keep_engineered = ["bmi", "pulse_pressure", "mean_arterial_pressure", "missing_count"]
        engineered = set(AGE_FEATURES + THRESHOLD_FEATURES + RATIO_FEATURES + ["bmi", "missing_count"])
        remove = [c for c in full.columns if c in engineered and c not in keep_engineered]
        return full.drop(columns=remove)
    raise ValueError(variant)


def make_model(frame: pd.DataFrame, c: float, gamma: float, epsilon: float) -> Pipeline:
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
            ("model", SVR(kernel="rbf", C=c, gamma=gamma, epsilon=epsilon, cache_size=1500)),
        ]
    )


def cv_score(
    features: pd.DataFrame,
    target: np.ndarray,
    seed: int,
    c: float,
    gamma: float,
    epsilon: float,
) -> tuple[float, list[float]]:
    oof = np.zeros(len(features), dtype=float)
    fold_scores: list[float] = []
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, valid_idx in cv.split(features):
        model = make_model(features, c, gamma, epsilon)
        model.fit(features.iloc[train_idx], target[train_idx])
        prediction = np.clip(model.predict(features.iloc[valid_idx]), 0.0, 1.0)
        oof[valid_idx] = prediction
        fold_scores.append(mean_absolute_error(target[valid_idx], prediction))
    return mean_absolute_error(target, oof), fold_scores


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    raw = train.drop(columns=[ID_COLUMN, TARGET])
    full_features = add_literature_features(raw, "physiology")
    target = train[TARGET].to_numpy(float)

    # Stage 1: 한 시드로 넓게 탐색한다. 이후 다른 시드가 독립 확인 역할을 한다.
    configs = [
        (c, gamma, epsilon)
        for c in (3.0, 5.0, 7.0, 10.0, 15.0)
        for gamma in (0.015, 0.022, 0.029, 0.040, 0.060)
        for epsilon in (0.0, 0.005, 0.01, 0.02)
    ]
    screen_rows = []
    print(f"Stage 1: {len(configs)}개 설정을 seed={SCREEN_SEED}에서 탐색합니다.")
    for index, (c, gamma, epsilon) in enumerate(configs, start=1):
        score, folds = cv_score(full_features, target, SCREEN_SEED, c, gamma, epsilon)
        screen_rows.append(
            {
                "C": c,
                "gamma": gamma,
                "epsilon": epsilon,
                "screen_MAE": score,
                "screen_fold_std": float(np.std(folds)),
            }
        )
        print(
            f"[Screen {index:03d}/{len(configs):03d}] C={c:<4} gamma={gamma:<5} "
            f"eps={epsilon:<5} | MAE={score:.6f}",
            flush=True,
        )

    screen = pd.DataFrame(screen_rows).sort_values(["screen_MAE", "screen_fold_std"])
    screen.to_csv(RESULT_DIR / "svr_v3_stage1_screen.csv", index=False)

    # 서로 너무 비슷한 후보까지 포함해 상위 8개를 독립 시드로 확인한다.
    finalists = screen.head(8)
    confirm_rows = []
    print("\nStage 2: 상위 8개를 3개 시드로 확인합니다.")
    for index, candidate in enumerate(finalists.itertuples(index=False), start=1):
        seed_scores = []
        fold_scores = []
        for seed in CONFIRM_SEEDS:
            score, folds = cv_score(
                full_features,
                target,
                seed,
                float(candidate.C),
                float(candidate.gamma),
                float(candidate.epsilon),
            )
            seed_scores.append(score)
            fold_scores.extend(folds)
        confirm_rows.append(
            {
                "C": candidate.C,
                "gamma": candidate.gamma,
                "epsilon": candidate.epsilon,
                "mean_MAE": float(np.mean(seed_scores)),
                "worst_seed_MAE": float(np.max(seed_scores)),
                "fold_std": float(np.std(fold_scores)),
                "seed_42": seed_scores[0],
                "seed_77": seed_scores[1],
                "seed_2026": seed_scores[2],
            }
        )
        print(
            f"[Confirm {index}/8] C={candidate.C} gamma={candidate.gamma} "
            f"eps={candidate.epsilon} | 평균={np.mean(seed_scores):.6f} | "
            f"최악={np.max(seed_scores):.6f}",
            flush=True,
        )

    confirm = pd.DataFrame(confirm_rows).sort_values(
        ["mean_MAE", "worst_seed_MAE", "fold_std"]
    ).reset_index(drop=True)
    confirm.to_csv(RESULT_DIR / "svr_v3_stage2_confirm.csv", index=False)
    best = confirm.iloc[0]

    # Stage 3: 최고 파라미터를 고정하고 피처군 제거 효과만 확인한다.
    variants = ("all", "no_age_interactions", "no_thresholds", "no_ratios", "core")
    ablation_rows = []
    print("\nStage 3: 피처 제거 검증을 진행합니다.")
    for variant in variants:
        features = select_features(full_features, variant)
        seed_scores = []
        fold_scores = []
        for seed in CONFIRM_SEEDS:
            score, folds = cv_score(
                features,
                target,
                seed,
                float(best["C"]),
                float(best["gamma"]),
                float(best["epsilon"]),
            )
            seed_scores.append(score)
            fold_scores.extend(folds)
        ablation_rows.append(
            {
                "variant": variant,
                "n_features_before_encoding": features.shape[1],
                "C": best["C"],
                "gamma": best["gamma"],
                "epsilon": best["epsilon"],
                "mean_MAE": float(np.mean(seed_scores)),
                "worst_seed_MAE": float(np.max(seed_scores)),
                "fold_std": float(np.std(fold_scores)),
                "seed_42": seed_scores[0],
                "seed_77": seed_scores[1],
                "seed_2026": seed_scores[2],
            }
        )
        print(
            f"[Ablation {variant:<20}] 평균={np.mean(seed_scores):.6f} | "
            f"최악={np.max(seed_scores):.6f} | 시드별={np.round(seed_scores, 6).tolist()}",
            flush=True,
        )

    ablation = pd.DataFrame(ablation_rows).sort_values(
        ["mean_MAE", "worst_seed_MAE", "fold_std"]
    ).reset_index(drop=True)
    ablation.to_csv(RESULT_DIR / "svr_v3_stage3_ablation.csv", index=False)
    winner = ablation.iloc[0]

    print("\n================ SVR V3 최종 결과 ================")
    print(f"최고 파라미터 : C={best['C']}, gamma={best['gamma']}, epsilon={best['epsilon']}")
    print(f"최고 피처군   : {winner['variant']}")
    print(f"평균 CV MAE   : {winner['mean_MAE']:.6f}")
    print(f"최악 시드 MAE : {winner['worst_seed_MAE']:.6f}")
    print(f"V2 대비 개선  : {0.196331 - winner['mean_MAE']:+.6f}")
    print(
        "시드별 MAE    : "
        f"42={winner['seed_42']:.6f}, 77={winner['seed_77']:.6f}, "
        f"2026={winner['seed_2026']:.6f}"
    )
    print("===================================================")


if __name__ == "__main__":
    main()
