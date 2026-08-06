from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
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
C = 5.0
ONEHOT_WEIGHT = 5.0


class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=20.0):
        self.smoothing = smoothing

    @staticmethod
    def _frame(values):
        frame = pd.DataFrame(values).copy()
        return frame.fillna("missing").astype(str)

    def fit(self, X, y):
        frame = self._frame(X)
        target = np.asarray(y, dtype=float)
        self.global_mean_ = float(np.mean(target))
        self.maps_ = []
        for column in frame.columns:
            stats = pd.DataFrame({"category": frame[column], "target": target}).groupby(
                "category"
            )["target"].agg(["mean", "count"])
            encoded = (
                stats["count"] * stats["mean"] + self.smoothing * self.global_mean_
            ) / (stats["count"] + self.smoothing)
            self.maps_.append(encoded.to_dict())
        return self

    def transform(self, X):
        frame = self._frame(X)
        columns = []
        for index, column in enumerate(frame.columns):
            columns.append(
                frame[column].map(self.maps_[index]).fillna(self.global_mean_).to_numpy(float)
            )
        return np.column_stack(columns)


def make_model(frame, gamma, target_weight, smoothing):
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
                "cat_onehot",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        (
                            "weight",
                            FunctionTransformer(
                                multiply_block,
                                kw_args={"weight": ONEHOT_WEIGHT},
                                validate=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
            (
                "cat_target",
                Pipeline(
                    [
                        ("encoder", SmoothedTargetEncoder(smoothing=smoothing)),
                        ("scaler", StandardScaler()),
                        (
                            "weight",
                            FunctionTransformer(
                                multiply_block,
                                kw_args={"weight": target_weight},
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
            ("model", SVR(C=C, gamma=gamma, epsilon=0.0, kernel="rbf", cache_size=1500)),
        ]
    )


def cv_score(features, target, seed, gamma, target_weight, smoothing):
    oof = np.zeros(len(features), dtype=float)
    folds = []
    for train_idx, valid_idx in KFold(5, shuffle=True, random_state=seed).split(features):
        model = make_model(features, gamma, target_weight, smoothing)
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
        (target_weight, smoothing, gamma)
        for target_weight in (0.0, 0.25, 0.50, 1.0, 2.0, 4.0)
        for smoothing in (10.0, 30.0)
        for gamma in (0.025, 0.035, 0.045)
    ]
    rows = []
    print(f"Stage 1: Fold-safe 타깃 인코딩 후보 {len(configs)}개를 탐색합니다.")
    for index, (weight, smoothing, gamma) in enumerate(configs, 1):
        score, folds = cv_score(features, target, 42, gamma, weight, smoothing)
        rows.append(
            {
                "target_weight": weight,
                "smoothing": smoothing,
                "onehot_weight": ONEHOT_WEIGHT,
                "C": C,
                "gamma": gamma,
                "screen_MAE": score,
                "fold_std": float(np.std(folds)),
            }
        )
        print(
            f"[{index:02d}/{len(configs):02d}] target_weight={weight:<4} "
            f"smooth={smoothing:<4} gamma={gamma:<5} MAE={score:.6f}",
            flush=True,
        )

    screen = pd.DataFrame(rows).sort_values(["screen_MAE", "fold_std"])
    screen_path = RESULT_DIR / "svr_v20_stage1_fold_target_encoding.csv"
    screen.to_csv(screen_path, index=False)
    finalists = screen.head(10)

    confirmed = []
    print("\nStage 2: 상위 10개 후보를 3개 시드에서 V19와 비교합니다.")
    for index, row in enumerate(finalists.itertuples(index=False), 1):
        scores, all_folds, improvements = [], [], []
        for seed in SEEDS:
            score, folds = cv_score(
                features,
                target,
                seed,
                float(row.gamma),
                float(row.target_weight),
                float(row.smoothing),
            )
            scores.append(score)
            all_folds.extend(folds)
            improvements.append(V19_BY_SEED[seed] - score)
        confirmed.append(
            {
                "target_weight": row.target_weight,
                "smoothing": row.smoothing,
                "onehot_weight": ONEHOT_WEIGHT,
                "C": C,
                "gamma": row.gamma,
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
            f"[Confirm {index:02d}/{len(finalists):02d}] target_weight={row.target_weight:<4} "
            f"smooth={row.smoothing:<4} gamma={row.gamma:<5} "
            f"평균={np.mean(scores):.6f} | 최악 개선={np.min(improvements):+.6f} | "
            f"V19 승리={sum(value > 0 for value in improvements)}/3",
            flush=True,
        )

    result = pd.DataFrame(confirmed).sort_values(
        ["mean_MAE", "worst_improvement_vs_v19", "fold_std"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    result_path = RESULT_DIR / "svr_v20_stage2_fold_target_confirm.csv"
    result.to_csv(result_path, index=False)
    best = result.iloc[0]

    print("\n================ SVR V20 최종 결과 ================")
    print(f"원-핫 가중치      : {ONEHOT_WEIGHT}")
    print(f"타깃 인코딩 가중치: {best['target_weight']}")
    print(f"스무딩            : {best['smoothing']}")
    print(f"최고 설정         : C={C}, gamma={best['gamma']}, epsilon=0")
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
