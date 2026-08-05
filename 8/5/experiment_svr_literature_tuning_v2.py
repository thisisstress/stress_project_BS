from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVR

from experiment_svr_literature_features import add_literature_features


DATA_DIR = Path("/Users/bitsaem/Desktop/stress_project/open")
RESULT_DIR = Path("/Users/bitsaem/Desktop/stress_project/8/5/8/5_result")
TARGET = "stress_score"
ID_COLUMN = "ID"
SEEDS = (42, 77, 2026)


def numerical_scaler(name: str, train_size: int):
    if name == "robust":
        return RobustScaler()
    if name == "standard":
        return StandardScaler()
    if name == "quantile":
        return QuantileTransformer(
            n_quantiles=min(500, train_size),
            output_distribution="normal",
            random_state=42,
        )
    raise ValueError(f"지원하지 않는 scaler: {name}")


def make_model(
    frame: pd.DataFrame,
    train_size: int,
    scaler_name: str,
    c: float,
    gamma: str | float,
    target_normal: bool,
) -> Pipeline:
    categorical = frame.select_dtypes(exclude="number").columns.tolist()
    numerical = frame.select_dtypes(include="number").columns.tolist()
    preprocessor = ColumnTransformer(
        [
            (
                "numerical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", numerical_scaler(scaler_name, train_size)),
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
    regressor = SVR(
        kernel="rbf",
        C=c,
        gamma=gamma,
        epsilon=0.01,
        cache_size=1500,
    )
    if target_normal:
        regressor = TransformedTargetRegressor(
            regressor=regressor,
            transformer=QuantileTransformer(
                n_quantiles=min(500, train_size),
                output_distribution="normal",
                random_state=42,
            ),
        )
    return Pipeline([("preprocessor", preprocessor), ("model", regressor)])


def evaluate_config(
    features: pd.DataFrame,
    target: np.ndarray,
    scaler_name: str,
    c: float,
    gamma: str | float,
    target_normal: bool,
) -> dict[str, object]:
    seed_scores: list[float] = []
    fold_scores: list[float] = []
    for seed in SEEDS:
        oof = np.zeros(len(features), dtype=float)
        cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        for train_idx, valid_idx in cv.split(features):
            model = make_model(
                features,
                len(train_idx),
                scaler_name,
                c,
                gamma,
                target_normal,
            )
            model.fit(features.iloc[train_idx], target[train_idx])
            prediction = np.clip(model.predict(features.iloc[valid_idx]), 0.0, 1.0)
            oof[valid_idx] = prediction
            fold_scores.append(mean_absolute_error(target[valid_idx], prediction))
        seed_scores.append(mean_absolute_error(target, oof))

    return {
        "scaler": scaler_name,
        "C": c,
        "gamma": gamma,
        "target_normal": target_normal,
        "mean_MAE": float(np.mean(seed_scores)),
        "worst_seed_MAE": float(np.max(seed_scores)),
        "fold_std": float(np.std(fold_scores)),
        "seed_42": seed_scores[0],
        "seed_77": seed_scores[1],
        "seed_2026": seed_scores[2],
    }


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    raw = train.drop(columns=[ID_COLUMN, TARGET])
    features = add_literature_features(raw, "physiology")
    target = train[TARGET].to_numpy(float)

    # 1차 결과(C=10, gamma=scale)를 중심으로 범위를 제한한다.
    configs = [
        (scaler, c, gamma, target_normal)
        for scaler in ("robust", "standard", "quantile")
        for c in (10.0, 30.0, 100.0)
        for gamma in ("scale",)
        for target_normal in (False, True)
    ]

    rows: list[dict[str, object]] = []
    total = len(configs)
    for index, (scaler, c, gamma, target_normal) in enumerate(configs, start=1):
        row = evaluate_config(features, target, scaler, c, gamma, target_normal)
        rows.append(row)
        print(
            f"[{index:02d}/{total:02d}] scaler={scaler:<8} C={c:<5} "
            f"target_normal={str(target_normal):<5} | "
            f"평균 MAE={row['mean_MAE']:.6f} | "
            f"최악 시드={row['worst_seed_MAE']:.6f} | "
            f"시드별={[round(row[f'seed_{s}'], 6) for s in SEEDS]}",
            flush=True,
        )

    result = pd.DataFrame(rows).sort_values(
        ["mean_MAE", "worst_seed_MAE", "fold_std"]
    ).reset_index(drop=True)
    output = RESULT_DIR / "svr_literature_tuning_v2_results.csv"
    result.to_csv(output, index=False)

    best = result.iloc[0]
    baseline = 0.204706
    print("\n========== SVR 논문 피처 V2 튜닝 결과 ==========")
    print(result.head(10).to_string(index=False))
    print("\n---------------- 최종 최고 결과 ----------------")
    print(f"Scaler        : {best['scaler']}")
    print(f"C / gamma      : {best['C']} / {best['gamma']}")
    print(f"Target normal  : {best['target_normal']}")
    print(f"평균 CV MAE    : {best['mean_MAE']:.6f}")
    print(f"최악 시드 MAE  : {best['worst_seed_MAE']:.6f}")
    print(f"V1 대비 개선   : {baseline - best['mean_MAE']:+.6f}")
    print(
        "시드별 MAE     : "
        f"42={best['seed_42']:.6f}, "
        f"77={best['seed_77']:.6f}, "
        f"2026={best['seed_2026']:.6f}"
    )
    print("=================================================")
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
