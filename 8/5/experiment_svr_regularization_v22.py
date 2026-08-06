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

from experiment_svr_block_weighting_v17 import multiply_block
from experiment_svr_v10_kernel_tuning_v11 import build_v10_winner


DATA_DIR = Path("/Users/bitsaem/Desktop/stress_project/open")
RESULT_DIR = Path("/Users/bitsaem/Desktop/stress_project/8/5/8/5_result")
TARGET = "stress_score"
ID_COLUMN = "ID"
SEEDS = (42, 77, 2026)
V19_BY_SEED = {
    42: 0.1575238167164372,
    77: 0.1564966653883252,
    2026: 0.1547984148591265,
}
CAT_WEIGHT = 5.0
GAMMA = 0.035


def make_model(frame, c, epsilon):
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
                                kw_args={"weight": CAT_WEIGHT},
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
                    gamma=GAMMA,
                    epsilon=epsilon,
                    kernel="rbf",
                    cache_size=1500,
                ),
            ),
        ]
    )


def cv_score(features, target, seed, c, epsilon):
    oof = np.zeros(len(features), dtype=float)
    folds = []
    for train_idx, valid_idx in KFold(5, shuffle=True, random_state=seed).split(features):
        model = make_model(features, c, epsilon)
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
        (c, epsilon)
        for c in (0.25, 0.50, 1.0, 2.0, 5.0, 10.0)
        for epsilon in (0.0, 0.0025, 0.005, 0.010, 0.020)
    ]
    rows = []
    print(f"Stage 1: V19 C·epsilon 후보 {len(configs)}개를 seed=42에서 탐색합니다.")
    for index, (c, epsilon) in enumerate(configs, 1):
        score, folds = cv_score(features, target, 42, c, epsilon)
        rows.append(
            {
                "C": c,
                "epsilon": epsilon,
                "categorical_weight": CAT_WEIGHT,
                "gamma": GAMMA,
                "screen_MAE": score,
                "fold_std": float(np.std(folds)),
            }
        )
        print(
            f"[{index:02d}/{len(configs):02d}] C={c:<4} epsilon={epsilon:<6} "
            f"MAE={score:.6f}",
            flush=True,
        )

    screen = pd.DataFrame(rows).sort_values(["screen_MAE", "fold_std"])
    screen_path = RESULT_DIR / "svr_v22_stage1_regularization.csv"
    screen.to_csv(screen_path, index=False)
    finalists = screen.head(10)

    confirmed = []
    print("\nStage 2: 상위 10개 후보를 3개 시드에서 V19와 비교합니다.")
    for index, row in enumerate(finalists.itertuples(index=False), 1):
        scores, all_folds, improvements = [], [], []
        for seed in SEEDS:
            score, folds = cv_score(
                features, target, seed, float(row.C), float(row.epsilon)
            )
            scores.append(score)
            all_folds.extend(folds)
            improvements.append(V19_BY_SEED[seed] - score)
        confirmed.append(
            {
                "C": row.C,
                "epsilon": row.epsilon,
                "categorical_weight": CAT_WEIGHT,
                "gamma": GAMMA,
                "mean_MAE": float(np.mean(scores)),
                "worst_seed_MAE": float(np.max(scores)),
                "fold_std": float(np.std(all_folds)),
                "seed_42": scores[0],
                "seed_77": scores[1],
                "seed_2026": scores[2],
                "mean_improvement_vs_v19": float(np.mean(improvements)),
                "worst_improvement_vs_v19": float(np.min(improvements)),
                "wins_vs_v19": int(sum(value > 0 for value in improvements)),
            }
        )
        print(
            f"[Confirm {index:02d}/{len(finalists):02d}] C={row.C:<4} "
            f"epsilon={row.epsilon:<6} 평균={np.mean(scores):.6f} | "
            f"최악 개선={np.min(improvements):+.6f} | "
            f"V19 승리={sum(value > 0 for value in improvements)}/3",
            flush=True,
        )

    result = pd.DataFrame(confirmed).sort_values(
        ["mean_MAE", "worst_improvement_vs_v19", "fold_std"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    result_path = RESULT_DIR / "svr_v22_stage2_regularization_confirm.csv"
    result.to_csv(result_path, index=False)
    best = result.iloc[0]

    print("\n================ SVR V22 최종 결과 ================")
    print(f"C / epsilon       : {best['C']} / {best['epsilon']}")
    print(f"가중치 / gamma    : {CAT_WEIGHT} / {GAMMA}")
    print(f"평균 CV MAE       : {best['mean_MAE']:.6f}")
    print(f"최악 시드 MAE     : {best['worst_seed_MAE']:.6f}")
    print(f"V19 대비 개선     : {best['mean_improvement_vs_v19']:+.6f}")
    print(f"최악 시드 개선    : {best['worst_improvement_vs_v19']:+.6f}")
    print(f"V19 승리 시드     : {int(best['wins_vs_v19'])}/3")
    print("====================================================")
    print(f"Screen results : {screen_path.resolve()}")
    print(f"Confirm results: {result_path.resolve()}")


if __name__ == "__main__":
    main()
