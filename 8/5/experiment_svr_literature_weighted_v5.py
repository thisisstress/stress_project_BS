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
V4_MAE = 0.169899


COPY_SCHEMES = {
    "none": {},
    "literature_light": {
        "bmi": 1,
        "pulse_pressure": 1,
        "mean_arterial_pressure": 1,
        "allostatic_count": 2,
        "age_allostatic": 1,
    },
    "literature_age": {
        "bmi": 1,
        "mean_arterial_pressure": 1,
        "allostatic_count": 2,
        "age_allostatic": 2,
        "age_bmi": 1,
        "age_map": 1,
        "age_glucose": 1,
        "age_bone_density": 1,
    },
    "tree_signal": {
        "mean_working": 1,
        "bmi": 4,
        "cholesterol": 2,
        "height": 2,
        "glucose": 3,
        "weight": 2,
        "cholesterol_glucose_ratio": 2,
        "bone_density": 4,
    },
    "balanced_signal": {
        "mean_working": 1,
        "bmi": 2,
        "cholesterol": 1,
        "height": 1,
        "glucose": 2,
        "weight": 1,
        "bone_density": 2,
        "mean_arterial_pressure": 1,
        "allostatic_count": 2,
        "age_allostatic": 1,
    },
}


def add_copies(frame: pd.DataFrame, scheme: str) -> pd.DataFrame:
    result = frame.copy()
    for column, count in COPY_SCHEMES[scheme].items():
        if column not in result:
            raise KeyError(f"{scheme}: {column} 피처가 없습니다.")
        for index in range(1, count + 1):
            result[f"{column}__weight_copy_{index}"] = result[column]
    return result


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

    configs = [
        (scheme, c, gamma)
        for scheme in COPY_SCHEMES
        for c in (0.7, 1.0, 2.0)
        for gamma in (0.06, 0.10, 0.14, 0.18)
    ]
    screen_rows = []
    print(f"Stage 1: 피처 가중치 후보 {len(configs)}개를 seed=42로 탐색합니다.")
    for index, (scheme, c, gamma) in enumerate(configs, 1):
        features = add_copies(base, scheme)
        score, folds = cv_score(features, target, 42, c, gamma)
        screen_rows.append(
            {
                "scheme": scheme,
                "C": c,
                "gamma": gamma,
                "screen_MAE": score,
                "fold_std": float(np.std(folds)),
                "n_input_features": features.shape[1],
            }
        )
        print(
            f"[{index:02d}/{len(configs):02d}] {scheme:<18} C={c:<3} gamma={gamma:<4} "
            f"MAE={score:.6f}",
            flush=True,
        )

    screen = pd.DataFrame(screen_rows).sort_values(["screen_MAE", "fold_std"])
    screen.to_csv(RESULT_DIR / "svr_v5_stage1_weight_screen.csv", index=False)

    # 동일 scheme만 몰리지 않도록 scheme별 최고 2개를 확인한다.
    finalists = (
        screen.groupby("scheme", group_keys=False)
        .head(2)
        .sort_values("screen_MAE")
        .head(10)
    )
    confirm_rows = []
    print("\nStage 2: scheme별 상위 후보를 3개 시드로 확인합니다.")
    for index, row in enumerate(finalists.itertuples(index=False), 1):
        features = add_copies(base, row.scheme)
        seed_scores = []
        fold_scores = []
        for seed in SEEDS:
            score, folds = cv_score(features, target, seed, float(row.C), float(row.gamma))
            seed_scores.append(score)
            fold_scores.extend(folds)
        confirm_rows.append(
            {
                "scheme": row.scheme,
                "C": row.C,
                "gamma": row.gamma,
                "mean_MAE": float(np.mean(seed_scores)),
                "worst_seed_MAE": float(np.max(seed_scores)),
                "fold_std": float(np.std(fold_scores)),
                "seed_42": seed_scores[0],
                "seed_77": seed_scores[1],
                "seed_2026": seed_scores[2],
                "wins_vs_v4": int(sum(s < V4_MAE for s in seed_scores)),
            }
        )
        print(
            f"[Confirm {index:02d}/{len(finalists):02d}] {row.scheme:<18} "
            f"평균={np.mean(seed_scores):.6f} 최악={np.max(seed_scores):.6f} "
            f"V4 승리={sum(s < V4_MAE for s in seed_scores)}/3",
            flush=True,
        )

    result = pd.DataFrame(confirm_rows).sort_values(
        ["mean_MAE", "worst_seed_MAE", "fold_std"]
    ).reset_index(drop=True)
    result_path = RESULT_DIR / "svr_v5_stage2_weight_confirm.csv"
    result.to_csv(result_path, index=False)
    best = result.iloc[0]
    print("\n================ SVR V5 최종 결과 ================")
    print(f"최고 가중치   : {best['scheme']}")
    print(f"최고 설정     : C={best['C']}, gamma={best['gamma']}, epsilon=0")
    print(f"평균 CV MAE   : {best['mean_MAE']:.6f}")
    print(f"최악 시드 MAE : {best['worst_seed_MAE']:.6f}")
    print(f"V4 대비 개선  : {V4_MAE - best['mean_MAE']:+.6f}")
    print(f"V4 승리 시드  : {int(best['wins_vs_v4'])}/3")
    print("===================================================")
    print(f"Saved: {result_path.resolve()}")


if __name__ == "__main__":
    main()
