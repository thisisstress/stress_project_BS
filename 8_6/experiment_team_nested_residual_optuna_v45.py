import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.svm import SVR
import optuna
import warnings

warnings.filterwarnings("ignore")

sys.path.append(str(Path("/Users/bitsaem/Desktop/stress_project/8/5")))
from experiment_team_agephys_selection_v38 import make_features
from experiment_team_feature_transfer_v35 import load_team_functions

DATA_DIR = Path("/Users/bitsaem/Desktop/stress_project/open")
RESULT_DIR = Path("/Users/bitsaem/Desktop/stress_project/8_6")
TARGET = "stress_score"
ID_COLUMN = "ID"
SEEDS = (42, 77, 2026)
AGE = {"age_bmi", "age_map", "age_glucose", "age_bone_density"}
GLUCOSE_BONE = {"age_glucose", "age_bone_density"}

def multiply_block(X, weight):
    return X * weight

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

def add_meta_columns(features, age_pred, gb_pred, use_poly=False):
    result = features.copy().reset_index(drop=True)
    result["age_pred"] = age_pred
    result["gb_pred"] = gb_pred
    result["v38_prediction"] = np.round(0.5 * age_pred + 0.5 * gb_pred, 2)
    result["age_gb_disagreement"] = np.abs(age_pred - gb_pred)
    
    if use_poly:
        result["age_pred_sq"] = age_pred ** 2
        result["gb_pred_sq"] = gb_pred ** 2
        result["pred_interaction"] = age_pred * gb_pred
    return result

def nested_cv_for_optuna(raw, y, seed, add_features, tree_predict, pair_predict, pair_weight, trial):
    # Optuna suggestions
    c = trial.suggest_float("C", 0.01, 10.0, log=True)
    gamma = trial.suggest_float("gamma", 0.001, 0.1, log=True)
    epsilon = trial.suggest_float("epsilon", 0.0, 0.05)
    alpha = trial.suggest_float("alpha", 0.1, 2.0)
    
    cat_weight = trial.suggest_float("cat_weight", 0.1, 10.0)
    num_weight = trial.suggest_float("num_weight", 0.1, 10.0)
    use_poly = trial.suggest_categorical("use_poly", [True, False])

    age_x = make_features(raw, add_features, AGE)
    gb_x = make_features(raw, add_features, GLUCOSE_BONE)
    meta_base = age_x.copy()
    
    outer = KFold(5, shuffle=True, random_state=seed)
    final_oof = np.zeros(len(raw))
    
    for fold, (outer_train, outer_valid) in enumerate(outer.split(raw)):
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

        meta_train = add_meta_columns(meta_base.iloc[outer_train], inner_age, inner_gb, use_poly)
        meta_valid = add_meta_columns(meta_base.iloc[outer_valid], valid_age, valid_gb, use_poly)
        residual = y.iloc[outer_train].to_numpy() - train_base
        
        categorical = meta_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        numerical = [column for column in meta_train.columns if column not in categorical]
        
        preprocessor = ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("weight", FunctionTransformer(multiply_block, kw_args={"weight": num_weight}, validate=False))
            ]), numerical),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ("weight", FunctionTransformer(multiply_block, kw_args={"weight": cat_weight}, validate=False))
            ]), categorical),
        ])
        
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("svr", SVR(kernel="rbf", C=c, gamma=gamma, epsilon=epsilon))
        ])
        
        model.fit(meta_train, residual)
        correction = model.predict(meta_valid)
        final_oof[outer_valid] = np.clip(valid_base + alpha * correction, 0, 1)

    return mean_absolute_error(y, final_oof)

def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    add_features, tree_predict, pair_predict, pair_weight = load_team_functions()
    train = pd.read_csv(DATA_DIR / "train.csv")
    raw = train.drop(columns=[ID_COLUMN, TARGET])
    y = train[TARGET].reset_index(drop=True)
    
    def objective(trial):
        # We optimize on seed 42 to save time during Optuna search
        return nested_cv_for_optuna(raw, y, 42, add_features, tree_predict, pair_predict, pair_weight, trial)
    
    study = optuna.create_study(direction="minimize")
    print("Starting Optuna search for Nested Residual SVR...")
    # Fast search with 30 trials
    study.optimize(objective, n_trials=30, show_progress_bar=True)
    
    print("\n================ Nested Optuna Finished ================")
    print(f"Best MAE on Seed 42: {study.best_value:.6f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
        
    results = study.trials_dataframe().sort_values("value")
    results.to_csv(RESULT_DIR / "team_v45_nested_svr_optuna.csv", index=False)

if __name__ == "__main__":
    main()
