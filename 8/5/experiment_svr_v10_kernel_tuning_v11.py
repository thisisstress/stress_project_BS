from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from experiment_svr_demographic_interactions_v6 import add_interactions, make_model
from experiment_svr_lifestyle_interactions_v10 import add_v10_features
from experiment_svr_literature_features import add_literature_features
from experiment_svr_positive_ordinal_v9 import build_variant


DATA_DIR = Path("/Users/bitsaem/Desktop/stress_project/open")
RESULT_DIR = Path("/Users/bitsaem/Desktop/stress_project/8/5/8/5_result")
TARGET = "stress_score"
ID_COLUMN = "ID"
SEEDS = (42, 77, 2026)
V10_BY_SEED = {42: 0.163288, 77: 0.161970, 2026: 0.161355}


def build_v10_winner(raw: pd.DataFrame) -> pd.DataFrame:
    physiology = add_literature_features(raw, "physiology")
    history = add_interactions(physiology, "history")
    ordinal = build_variant(history, "education_activity_smoking")
    return add_v10_features(ordinal, "lifestyle_plus_sex")


def cv_score(
    features: pd.DataFrame,
    target: np.ndarray,
    seed: int,
    c: float,
    gamma: float,
    return_oof: bool = False,
):
    oof = np.zeros(len(features), dtype=float)
    fold_scores = []
    splitter = KFold(5, shuffle=True, random_state=seed)
    for train_idx, valid_idx in splitter.split(features):
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
    features = build_v10_winner(raw)
    target = train[TARGET].to_numpy(float)

    c_values = (0.7, 1.0, 1.5, 2.0, 3.0, 5.0)
    gamma_values = (0.04, 0.05, 0.06, 0.07, 0.08, 0.09)
    configs = [(c, gamma) for c in c_values for gamma in gamma_values]

    screen_rows = []
    print(f"Stage 1: V10 피처를 고정하고 커널 후보 {len(configs)}개를 seed=42에서 탐색합니다.")
    for index, (c, gamma) in enumerate(configs, 1):
        score, folds = cv_score(features, target, 42, c, gamma)
        screen_rows.append(
            {
                "C": c,
                "gamma": gamma,
                "screen_MAE": score,
                "fold_std": float(np.std(folds)),
            }
        )
        print(
            f"[Screen {index:02d}/{len(configs):02d}] C={c:<3} gamma={gamma:<4} "
            f"MAE={score:.6f}",
            flush=True,
        )

    screen = pd.DataFrame(screen_rows).sort_values(["screen_MAE", "fold_std"])
    screen_path = RESULT_DIR / "svr_v11_stage1_kernel_screen.csv"
    screen.to_csv(screen_path, index=False)

    # 한 시드에만 맞춘 후보를 피하기 위해 상위 10개를 독립 시드 3개로 재검증한다.
    finalists = screen.head(10)
    confirm_rows = []
    print("\nStage 2: 상위 10개 후보를 동일한 3개 시드에서 V10과 직접 비교합니다.")
    for index, row in enumerate(finalists.itertuples(index=False), 1):
        seed_scores = []
        all_folds = []
        improvements = []
        for seed in SEEDS:
            score, folds = cv_score(features, target, seed, float(row.C), float(row.gamma))
            seed_scores.append(score)
            all_folds.extend(folds)
            improvements.append(V10_BY_SEED[seed] - score)
        confirm_rows.append(
            {
                "C": row.C,
                "gamma": row.gamma,
                "mean_MAE": float(np.mean(seed_scores)),
                "worst_seed_MAE": float(np.max(seed_scores)),
                "fold_std": float(np.std(all_folds)),
                "seed_42": seed_scores[0],
                "seed_77": seed_scores[1],
                "seed_2026": seed_scores[2],
                "mean_improvement_vs_v10": float(np.mean(improvements)),
                "worst_improvement_vs_v10": float(np.min(improvements)),
                "wins_vs_v10": int(sum(value > 0 for value in improvements)),
            }
        )
        print(
            f"[Confirm {index:02d}/{len(finalists):02d}] C={row.C:<3} gamma={row.gamma:<4} "
            f"평균={np.mean(seed_scores):.6f} | 최악 개선={np.min(improvements):+.6f} | "
            f"V10 승리={sum(value > 0 for value in improvements)}/3",
            flush=True,
        )

    result = pd.DataFrame(confirm_rows).sort_values(
        ["mean_MAE", "worst_improvement_vs_v10", "fold_std"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    confirm_path = RESULT_DIR / "svr_v11_stage2_kernel_confirm.csv"
    result.to_csv(confirm_path, index=False)
    best = result.iloc[0]

    # 최종 후보의 시드별 OOF를 남겨 이후 잔차 분석과 블렌딩을 누수 없이 수행한다.
    oof_frame = pd.DataFrame({ID_COLUMN: train[ID_COLUMN], TARGET: target})
    for seed in SEEDS:
        _, _, oof = cv_score(
            features,
            target,
            seed,
            float(best["C"]),
            float(best["gamma"]),
            return_oof=True,
        )
        oof_frame[f"pred_seed_{seed}"] = oof
    oof_frame["pred_mean"] = oof_frame[[f"pred_seed_{s}" for s in SEEDS]].mean(axis=1)
    oof_path = RESULT_DIR / "svr_v11_best_oof.csv"
    oof_frame.to_csv(oof_path, index=False)

    print("\n================ SVR V11 최종 결과 ================")
    print(f"최고 설정       : C={best['C']}, gamma={best['gamma']}, epsilon=0")
    print(f"평균 CV MAE     : {best['mean_MAE']:.6f}")
    print(f"최악 시드 MAE   : {best['worst_seed_MAE']:.6f}")
    print(f"V10 대비 개선   : {best['mean_improvement_vs_v10']:+.6f}")
    print(f"최악 시드 개선  : {best['worst_improvement_vs_v10']:+.6f}")
    print(f"V10 승리 시드   : {int(best['wins_vs_v10'])}/3")
    print("====================================================")
    print(f"Screen results : {screen_path.resolve()}")
    print(f"Confirm results: {confirm_path.resolve()}")
    print(f"Best OOF       : {oof_path.resolve()}")


if __name__ == "__main__":
    main()
