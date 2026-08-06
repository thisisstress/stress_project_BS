from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.svm import SVR

from experiment_svr_block_weighting_v17 import multiply_block
from experiment_svr_v10_kernel_tuning_v11 import build_v10_winner


DATA_DIR = Path("/Users/bitsaem/Desktop/stress_project/open")
RESULT_DIR = Path("/Users/bitsaem/Desktop/stress_project/8/5/8/5_result")
TARGET = "stress_score"
ID_COLUMN = "ID"
SEEDS = (42, 77, 2026)
V26_BY_SEED = {
    42: 0.1560325877080313,
    77: 0.1546974842882678,
    2026: 0.1535492286071821,
}
C = 5.0
N_BINS = 4
BIN_WEIGHT = 1.3
CAT_WEIGHT = 7.0

# 구간화가 의미 있는 연속형 건강 변수만 사용합니다. 이진 플래그, 결측치
# 표시, ordinal 값, 피처 가중치용 복제 열까지 binning하면 동일 경계가
# 반복되어 KBins 경고가 쏟아지고 불필요한 차원만 늘어납니다.
BINNABLE_COLUMNS = (
    "age",
    "height",
    "weight",
    "cholesterol",
    "systolic_blood_pressure",
    "diastolic_blood_pressure",
    "glucose",
    "bone_density",
    "bmi",
    "pulse_pressure",
    "mean_arterial_pressure",
    "bp_ratio",
    "cholesterol_glucose_ratio",
    "metabolic_product",
    "age_bmi",
    "age_map",
    "age_glucose",
    "age_bone_density",
)

BIN_GROUPS = {
    "anthropometric": {"age", "height", "weight", "bmi", "age_bmi"},
    "blood_pressure": {
        "systolic_blood_pressure", "diastolic_blood_pressure", "pulse_pressure",
        "mean_arterial_pressure", "bp_ratio", "age_map",
    },
    "metabolic": {
        "cholesterol", "glucose", "cholesterol_glucose_ratio",
        "metabolic_product", "age_glucose",
    },
    "bone": {"bone_density", "age_bone_density"},
}


def make_model(frame, gamma, selected_binnable):
    categorical = frame.select_dtypes(exclude="number").columns.tolist()
    numerical = frame.select_dtypes(include="number").columns.tolist()
    binnable = [column for column in selected_binnable if column in numerical]
    if not binnable:
        raise ValueError("구간화할 연속형 피처를 찾지 못했습니다.")
    preprocessor = ColumnTransformer(
        [
            (
                "num_continuous",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical,
            ),
            (
                "num_binned",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "bins",
                            KBinsDiscretizer(
                                n_bins=N_BINS,
                                encode="onehot-dense",
                                strategy="quantile",
                                subsample=None,
                            ),
                        ),
                        (
                            "weight",
                            FunctionTransformer(
                                multiply_block,
                                kw_args={"weight": BIN_WEIGHT},
                                validate=False,
                            ),
                        ),
                    ]
                ),
                binnable,
            ),
            (
                "categorical",
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
            ("model", SVR(C=C, gamma=gamma, epsilon=0.0, kernel="rbf", cache_size=1500)),
        ]
    )


def cv_score(features, target, seed, gamma, selected_binnable):
    oof = np.zeros(len(features), dtype=float)
    folds = []
    for train_idx, valid_idx in KFold(5, shuffle=True, random_state=seed).split(features):
        model = make_model(features, gamma, selected_binnable)
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

    full = set(BINNABLE_COLUMNS)
    selections = {"full": full}
    for name, group in BIN_GROUPS.items():
        selections[f"drop_{name}"] = full - group
        selections[f"only_{name}"] = group
    selections.update({
        "anthro_bp": BIN_GROUPS["anthropometric"] | BIN_GROUPS["blood_pressure"],
        "anthro_metabolic": BIN_GROUPS["anthropometric"] | BIN_GROUPS["metabolic"],
        "bp_metabolic": BIN_GROUPS["blood_pressure"] | BIN_GROUPS["metabolic"],
        "core_three": full - BIN_GROUPS["bone"],
    })
    configs = [
        (name, tuple(sorted(columns)), gamma)
        for name, columns in selections.items()
        for gamma in (0.014, 0.018, 0.022)
    ]
    rows = []
    print(f"Stage 1: 구간 피처 묶음 제거 후보 {len(configs)}개를 탐색합니다.")
    for index, (selection, columns, gamma) in enumerate(configs, 1):
        score, folds = cv_score(features, target, 42, gamma, columns)
        rows.append(
            {
                "selection": selection,
                "binned_columns": "|".join(columns),
                "binned_feature_count": len(columns),
                "C": C,
                "gamma": gamma,
                "screen_MAE": score,
                "fold_std": float(np.std(folds)),
            }
        )
        print(
            f"[{index:02d}/{len(configs):02d}] {selection:<20} "
            f"features={len(columns):<2} gamma={gamma:<5} MAE={score:.6f}",
            flush=True,
        )

    screen = pd.DataFrame(rows).sort_values(["screen_MAE", "fold_std"])
    screen_path = RESULT_DIR / "svr_v28_stage1_binned_group_ablation.csv"
    screen.to_csv(screen_path, index=False)
    finalists = screen.groupby("selection", as_index=False).first().sort_values("screen_MAE").head(8)

    confirmed = []
    print("\nStage 2: 피처 구성별 상위 8개를 3개 시드에서 V26과 비교합니다.")
    for index, row in enumerate(finalists.itertuples(index=False), 1):
        scores, all_folds, improvements = [], [], []
        for seed in SEEDS:
            columns = tuple(str(row.binned_columns).split("|"))
            score, folds = cv_score(features, target, seed, float(row.gamma), columns)
            scores.append(score)
            all_folds.extend(folds)
            improvements.append(V26_BY_SEED[seed] - score)
        confirmed.append(
            {
                "selection": row.selection,
                "binned_columns": row.binned_columns,
                "binned_feature_count": row.binned_feature_count,
                "C": C,
                "gamma": row.gamma,
                "mean_MAE": float(np.mean(scores)),
                "worst_seed_MAE": float(np.max(scores)),
                "fold_std": float(np.std(all_folds)),
                "seed_42": scores[0],
                "seed_77": scores[1],
                "seed_2026": scores[2],
                "mean_improvement_vs_v26": float(np.mean(improvements)),
                "worst_improvement_vs_v26": float(np.min(improvements)),
                "wins_vs_v26": int(sum(value > 0 for value in improvements)),
            }
        )
        print(
            f"[Confirm {index:02d}/{len(finalists):02d}] {row.selection:<20} "
            f"features={int(row.binned_feature_count):<2} gamma={row.gamma:<5} "
            f"평균={np.mean(scores):.6f} | 최악 개선={np.min(improvements):+.6f} | "
            f"V26 승리={sum(value > 0 for value in improvements)}/3",
            flush=True,
        )

    result = pd.DataFrame(confirmed).sort_values(
        ["mean_MAE", "worst_improvement_vs_v26", "fold_std"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    result_path = RESULT_DIR / "svr_v28_stage2_binned_group_confirm.csv"
    result.to_csv(result_path, index=False)
    best = result.iloc[0]

    print("\n================ SVR V28 최종 결과 ================")
    print(f"구간 피처 구성   : {best['selection']}")
    print(f"구간 피처 수     : {int(best['binned_feature_count'])}")
    print(f"구간/범주 가중치: {BIN_WEIGHT} / {CAT_WEIGHT}")
    print(f"C / gamma        : {C} / {best['gamma']}")
    print(f"평균 CV MAE      : {best['mean_MAE']:.6f}")
    print(f"최악 시드 MAE    : {best['worst_seed_MAE']:.6f}")
    print(f"V26 대비 개선    : {best['mean_improvement_vs_v26']:+.6f}")
    print(f"최악 시드 개선   : {best['worst_improvement_vs_v26']:+.6f}")
    print(f"V26 승리 시드    : {int(best['wins_vs_v26'])}/3")
    print("====================================================")
    print(f"Screen results : {screen_path.resolve()}")
    print(f"Confirm results: {result_path.resolve()}")


if __name__ == "__main__":
    main()
