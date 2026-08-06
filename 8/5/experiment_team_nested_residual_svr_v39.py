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

from experiment_team_agephys_selection_v38 import make_features
from experiment_team_feature_transfer_v35 import (
    DATA_DIR, ID_COLUMN, RESULT_DIR, SEEDS, TARGET, load_team_functions,
)


AGE = {"age_bmi", "age_map", "age_glucose", "age_bone_density"}
GLUCOSE_BONE = {"age_glucose", "age_bone_density"}
CONFIGS = [
    (0.10, "scale", 0.00),
    (0.25, "scale", 0.00),
    (0.50, "scale", 0.00),
    (1.00, "scale", 0.00),
    (2.00, "scale", 0.00),
    (0.50, 0.01, 0.00),
    (1.00, 0.01, 0.00),
    (2.00, 0.01, 0.00),
]
ALPHAS = (0.25, 0.50, 0.75, 1.00)


def component_prediction(train_x, valid_x, train_y, tree_predict, pair_predict, pair_weight):
    tree = tree_predict(train_x, valid_x, train_y)
    pair = pair_predict(train_x, valid_x, train_y)
    return np.clip(np.round((1 - pair_weight) * tree + pair_weight * pair, 2), 0, 1)


def v38_components(age_x, gb_x, y, train_idx, valid_idx, tree_predict, pair_predict, pair_weight):
    age = component_prediction(
        age_x.iloc[train_idx], age_x.iloc[valid_idx], y.iloc[train_idx],
        tree_predict, pair_predict, pair_weight,
    )
    gb = component_prediction(
        gb_x.iloc[train_idx], gb_x.iloc[valid_idx], y.iloc[train_idx],
        tree_predict, pair_predict, pair_weight,
    )
    return age, gb


def add_meta_columns(features, age_pred, gb_pred):
    result = features.copy().reset_index(drop=True)
    result["v38_prediction"] = np.round(0.5 * age_pred + 0.5 * gb_pred, 2)
    result["age_gb_disagreement"] = np.abs(age_pred - gb_pred)
    result["age_gb_mean_raw"] = 0.5 * age_pred + 0.5 * gb_pred
    return result


def make_residual_model(frame, c, gamma, epsilon):
    categorical = frame.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numerical = [column for column in frame.columns if column not in categorical]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numerical),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("svr", SVR(kernel="rbf", C=c, gamma=gamma, epsilon=epsilon)),
    ])


def nested_seed(raw, y, seed, add_features, tree_predict, pair_predict, pair_weight):
    age_x = make_features(raw, add_features, AGE)
    gb_x = make_features(raw, add_features, GLUCOSE_BONE)
    # SVR에는 검증된 연령 생리 피처와 기존 팀 피처만 전달한다.
    meta_base = age_x.copy()
    outer = KFold(5, shuffle=True, random_state=seed)
    baseline_oof = np.zeros(len(raw))
    candidate_oof = {config: {alpha: np.zeros(len(raw)) for alpha in ALPHAS} for config in CONFIGS}

    for fold, (outer_train, outer_valid) in enumerate(outer.split(raw), 1):
        inner_age = np.zeros(len(outer_train))
        inner_gb = np.zeros(len(outer_train))
        inner = KFold(4, shuffle=True, random_state=seed + 1000 + fold)
        for inner_train_pos, inner_valid_pos in inner.split(outer_train):
            inner_train = outer_train[inner_train_pos]
            inner_valid = outer_train[inner_valid_pos]
            age_pred, gb_pred = v38_components(
                age_x, gb_x, y, inner_train, inner_valid,
                tree_predict, pair_predict, pair_weight,
            )
            inner_age[inner_valid_pos] = age_pred
            inner_gb[inner_valid_pos] = gb_pred

        valid_age, valid_gb = v38_components(
            age_x, gb_x, y, outer_train, outer_valid,
            tree_predict, pair_predict, pair_weight,
        )
        train_base = np.round(0.5 * inner_age + 0.5 * inner_gb, 2)
        valid_base = np.round(0.5 * valid_age + 0.5 * valid_gb, 2)
        baseline_oof[outer_valid] = valid_base

        meta_train = add_meta_columns(
            meta_base.iloc[outer_train], inner_age, inner_gb
        )
        meta_valid = add_meta_columns(
            meta_base.iloc[outer_valid], valid_age, valid_gb
        )
        residual = y.iloc[outer_train].to_numpy() - train_base
        for config in CONFIGS:
            c, gamma, epsilon = config
            model = make_residual_model(meta_train, c, gamma, epsilon)
            model.fit(meta_train, residual)
            correction = model.predict(meta_valid)
            for alpha in ALPHAS:
                candidate_oof[config][alpha][outer_valid] = np.clip(
                    valid_base + alpha * correction, 0, 1
                )
        print(f"Seed {seed} | Outer Fold {fold}/5 complete", flush=True)
    return baseline_oof, candidate_oof


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    add_features, tree_predict, pair_predict, pair_weight = load_team_functions()
    train = pd.read_csv(DATA_DIR / "train.csv")
    raw = train.drop(columns=[ID_COLUMN, TARGET])
    y = train[TARGET].reset_index(drop=True)

    all_results = {}
    for seed in SEEDS:
        print(f"\nNested residual SVR seed={seed}를 시작합니다.")
        all_results[seed] = nested_seed(
            raw, y, seed, add_features, tree_predict, pair_predict, pair_weight
        )

    rows = []
    for config in CONFIGS:
        for alpha in ALPHAS:
            for round2 in (False, True):
                scores, gains = [], []
                for seed in SEEDS:
                    baseline, candidates = all_results[seed]
                    prediction = candidates[config][alpha]
                    if round2:
                        prediction = np.round(prediction, 2)
                    base_score = mean_absolute_error(y, baseline)
                    score = mean_absolute_error(y, prediction)
                    scores.append(score)
                    gains.append(base_score - score)
                rows.append({
                    "C": config[0], "gamma": config[1], "epsilon": config[2],
                    "residual_weight": alpha, "round2": round2,
                    "mean_MAE": np.mean(scores), "worst_seed_MAE": np.max(scores),
                    "seed_42": scores[0], "seed_77": scores[1], "seed_2026": scores[2],
                    "mean_improvement": np.mean(gains),
                    "worst_improvement": np.min(gains),
                    "wins": sum(g > 0 for g in gains),
                })

    result = pd.DataFrame(rows).sort_values(
        ["wins", "worst_improvement", "mean_MAE"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    path = RESULT_DIR / "team_v39_nested_residual_svr.csv"
    result.to_csv(path, index=False)
    stable = result.iloc[0]
    eligible = result[result.wins == 3]
    aggressive = eligible.sort_values("mean_MAE").iloc[0] if len(eligible) else stable

    print("\n================ Team V39 최종 결과 ================")
    print(f"안정성 설정       : C={stable.C}, gamma={stable.gamma}, residual={stable.residual_weight}")
    print(f"반올림            : {bool(stable.round2)}")
    print(f"평균 CV MAE       : {stable.mean_MAE:.6f}")
    print(f"최악 시드 개선    : {stable.worst_improvement:+.6f}")
    print(f"승리 시드         : {int(stable.wins)}/3")
    print("-----------------------------------------------------")
    print(f"평균 최저 설정    : C={aggressive.C}, gamma={aggressive.gamma}, residual={aggressive.residual_weight}")
    print(f"최저 평균 CV MAE  : {aggressive.mean_MAE:.6f}")
    print(f"최악 시드 개선    : {aggressive.worst_improvement:+.6f}")
    print("=====================================================")
    print(f"Saved: {Path(path).resolve()}")


if __name__ == "__main__":
    main()
