from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVR

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


def multiply_block(values, weight=1.0):
    return values * weight


def make_block_weighted_model(frame, c, gamma, categorical_weight):
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
                        (
                            "weight",
                            FunctionTransformer(
                                multiply_block,
                                kw_args={"weight": categorical_weight},
                                validate=False,
                            ),
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
                SVR(
                    C=c,
                    gamma=gamma,
                    epsilon=0.0,
                    kernel="rbf",
                    cache_size=1500,
                ),
            ),
        ]
    )


def cv_score(features, target, seed, c, gamma, categorical_weight):
    oof = np.zeros(len(features), dtype=float)
    folds = []
    for train_idx, valid_idx in KFold(5, shuffle=True, random_state=seed).split(features):
        model = make_block_weighted_model(features, c, gamma, categorical_weight)
        model.fit(features.iloc[train_idx], target[train_idx])
        pred = np.clip(model.predict(features.iloc[valid_idx]), 0.0, 1.0)
        oof[valid_idx] = pred
        folds.append(mean_absolute_error(target[valid_idx], pred))
    return mean_absolute_error(target, oof), folds


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    features = build_v10_winner(train.drop(columns=[ID_COLUMN, TARGET]))
    target = train[TARGET].to_numpy(float)

    configs = [
        (weight, c, gamma)
        for weight in (0.25, 0.40, 0.60, 0.80, 1.00, 1.25, 1.50)
        for c in (2.0, 5.0)
        for gamma in (0.04, 0.08, 0.12)
    ]
    screen_rows = []
    print(f"Stage 1: V11 범주형 거리 가중치 후보 {len(configs)}개를 탐색합니다.")
    for index, (weight, c, gamma) in enumerate(configs, 1):
        score, folds = cv_score(features, target, 42, c, gamma, weight)
        screen_rows.append(
            {
                "categorical_weight": weight,
                "C": c,
                "gamma": gamma,
                "screen_MAE": score,
                "fold_std": float(np.std(folds)),
            }
        )
        print(
            f"[{index:02d}/{len(configs):02d}] cat_weight={weight:<4} "
            f"C={c} gamma={gamma} MAE={score:.6f}",
            flush=True,
        )

    screen = pd.DataFrame(screen_rows).sort_values(["screen_MAE", "fold_std"])
    screen_path = RESULT_DIR / "svr_v17_stage1_block_weighting.csv"
    screen.to_csv(screen_path, index=False)
    finalists = screen.groupby("categorical_weight", group_keys=False).head(1).sort_values(
        "screen_MAE"
    )

    rows = []
    print("\nStage 2: 가중치별 최고 후보를 3개 시드에서 V11과 비교합니다.")
    for index, row in enumerate(finalists.itertuples(index=False), 1):
        scores, all_folds, improvements = [], [], []
        for seed in SEEDS:
            score, folds = cv_score(
                features,
                target,
                seed,
                float(row.C),
                float(row.gamma),
                float(row.categorical_weight),
            )
            scores.append(score)
            all_folds.extend(folds)
            improvements.append(V11_BY_SEED[seed] - score)
        rows.append(
            {
                "categorical_weight": row.categorical_weight,
                "C": row.C,
                "gamma": row.gamma,
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
            f"[Confirm {index}/{len(finalists)}] cat_weight={row.categorical_weight:<4} "
            f"평균={np.mean(scores):.6f} | 최악 개선={np.min(improvements):+.6f} | "
            f"V11 승리={sum(value > 0 for value in improvements)}/3",
            flush=True,
        )

    result = pd.DataFrame(rows).sort_values(
        ["mean_MAE", "worst_improvement_vs_v11", "fold_std"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    result_path = RESULT_DIR / "svr_v17_stage2_block_weighting_confirm.csv"
    result.to_csv(result_path, index=False)
    best = result.iloc[0]

    print("\n================ SVR V17 최종 결과 ================")
    print(f"범주형 거리 가중치: {best['categorical_weight']}")
    print(f"최고 설정         : C={best['C']}, gamma={best['gamma']}, epsilon=0")
    print(f"평균 CV MAE       : {best['mean_MAE']:.6f}")
    print(f"최악 시드 MAE     : {best['worst_seed_MAE']:.6f}")
    print(f"V11 대비 개선     : {best['mean_improvement_vs_v11']:+.6f}")
    print(f"최악 시드 개선    : {best['worst_improvement_vs_v11']:+.6f}")
    print(f"V11 승리 시드     : {int(best['wins_vs_v11'])}/3")
    print("====================================================")
    print(f"Screen results : {screen_path.resolve()}")
    print(f"Confirm results: {result_path.resolve()}")


if __name__ == "__main__":
    main()
