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
V3_MAE = 0.179315


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
            (
                "model",
                SVR(kernel="rbf", C=c, gamma=gamma, epsilon=0.0, cache_size=1500),
            ),
        ]
    )


def run_cv(
    features: pd.DataFrame,
    target: np.ndarray,
    seed: int,
    c: float,
    gamma: float,
    return_oof: bool = False,
):
    oof = np.zeros(len(features), dtype=float)
    fold_scores = []
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, valid_idx in cv.split(features):
        model = make_model(features, c, gamma)
        model.fit(features.iloc[train_idx], target[train_idx])
        prediction = np.clip(model.predict(features.iloc[valid_idx]), 0.0, 1.0)
        oof[valid_idx] = prediction
        fold_scores.append(mean_absolute_error(target[valid_idx], prediction))
    score = mean_absolute_error(target, oof)
    if return_oof:
        return score, fold_scores, oof
    return score, fold_scores


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    raw = train.drop(columns=[ID_COLUMN, TARGET])
    features = add_literature_features(raw, "physiology")
    target = train[TARGET].to_numpy(float)

    c_values = (0.3, 0.5, 1.0, 2.0, 3.0, 4.0)
    gamma_values = (0.06, 0.08, 0.10, 0.13, 0.16, 0.20)
    configs = [(c, gamma) for c in c_values for gamma in gamma_values]

    screen_rows = []
    print(f"Stage 1: {len(configs)}개 설정을 seed={SCREEN_SEED}에서 탐색합니다.")
    for index, (c, gamma) in enumerate(configs, start=1):
        score, folds = run_cv(features, target, SCREEN_SEED, c, gamma)
        screen_rows.append(
            {
                "C": c,
                "gamma": gamma,
                "epsilon": 0.0,
                "screen_MAE": score,
                "screen_fold_std": float(np.std(folds)),
            }
        )
        print(
            f"[Screen {index:02d}/{len(configs):02d}] C={c:<3} gamma={gamma:<4} | "
            f"MAE={score:.6f}",
            flush=True,
        )

    screen = pd.DataFrame(screen_rows).sort_values(["screen_MAE", "screen_fold_std"])
    screen_path = RESULT_DIR / "svr_v4_stage1_screen.csv"
    screen.to_csv(screen_path, index=False)

    finalists = screen.head(8)
    confirm_rows = []
    oof_by_candidate: dict[tuple[float, float], dict[int, np.ndarray]] = {}
    print("\nStage 2: 상위 8개를 3개 시드에서 재검증합니다.")
    for index, candidate in enumerate(finalists.itertuples(index=False), start=1):
        c = float(candidate.C)
        gamma = float(candidate.gamma)
        seed_scores = []
        fold_scores = []
        oof_by_candidate[(c, gamma)] = {}
        for seed in CONFIRM_SEEDS:
            score, folds, oof = run_cv(
                features, target, seed, c, gamma, return_oof=True
            )
            seed_scores.append(score)
            fold_scores.extend(folds)
            oof_by_candidate[(c, gamma)][seed] = oof
        confirm_rows.append(
            {
                "C": c,
                "gamma": gamma,
                "epsilon": 0.0,
                "mean_MAE": float(np.mean(seed_scores)),
                "worst_seed_MAE": float(np.max(seed_scores)),
                "fold_std": float(np.std(fold_scores)),
                "seed_42": seed_scores[0],
                "seed_77": seed_scores[1],
                "seed_2026": seed_scores[2],
                "wins_vs_v3": int(sum(score < V3_MAE for score in seed_scores)),
            }
        )
        print(
            f"[Confirm {index}/8] C={c} gamma={gamma} | "
            f"평균={np.mean(seed_scores):.6f} | 최악={np.max(seed_scores):.6f} | "
            f"V3 승리={sum(score < V3_MAE for score in seed_scores)}/3",
            flush=True,
        )

    confirm = pd.DataFrame(confirm_rows).sort_values(
        ["mean_MAE", "worst_seed_MAE", "fold_std"]
    ).reset_index(drop=True)
    confirm_path = RESULT_DIR / "svr_v4_stage2_confirm.csv"
    confirm.to_csv(confirm_path, index=False)
    best = confirm.iloc[0]

    best_key = (float(best["C"]), float(best["gamma"]))
    oof_output = pd.DataFrame({ID_COLUMN: train[ID_COLUMN], TARGET: target})
    for seed in CONFIRM_SEEDS:
        oof_output[f"oof_seed_{seed}"] = oof_by_candidate[best_key][seed]
    oof_output.to_csv(RESULT_DIR / "svr_v4_best_oof.csv", index=False)

    print("\n================ SVR V4 최종 결과 ================")
    print(f"최고 설정     : C={best['C']}, gamma={best['gamma']}, epsilon=0")
    print(f"평균 CV MAE   : {best['mean_MAE']:.6f}")
    print(f"최악 시드 MAE : {best['worst_seed_MAE']:.6f}")
    print(f"V3 대비 개선  : {V3_MAE - best['mean_MAE']:+.6f}")
    print(f"V3 승리 시드  : {int(best['wins_vs_v3'])}/3")
    print(
        "시드별 MAE    : "
        f"42={best['seed_42']:.6f}, 77={best['seed_77']:.6f}, "
        f"2026={best['seed_2026']:.6f}"
    )
    print("===================================================")
    print(f"Screen results : {screen_path.resolve()}")
    print(f"Confirm results: {confirm_path.resolve()}")


if __name__ == "__main__":
    main()
