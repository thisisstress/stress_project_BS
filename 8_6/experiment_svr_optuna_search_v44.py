import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
import optuna
import warnings

warnings.filterwarnings("ignore")

# Add the parent directory so we can import the user's existing feature modules
sys.path.append(str(Path("/Users/bitsaem/Desktop/stress_project/8/5")))
from experiment_svr_v10_kernel_tuning_v11 import build_v10_winner

DATA_DIR = Path("/Users/bitsaem/Desktop/stress_project/open")
RESULT_DIR = Path("/Users/bitsaem/Desktop/stress_project/8_6")
TARGET = "stress_score"
ID_COLUMN = "ID"

def multiply_block(X, weight):
    return X * weight

def objective(trial, features, target):
    # 1. SVR Hyperparameters
    c = trial.suggest_float("C", 0.1, 50.0, log=True)
    gamma = trial.suggest_float("gamma", 0.001, 0.2, log=True)
    epsilon = trial.suggest_float("epsilon", 0.0, 0.05)
    
    # 2. Feature Weights
    cat_weight = trial.suggest_float("cat_weight", 0.1, 15.0)
    num_weight = trial.suggest_float("num_weight", 0.1, 5.0)
    
    # Optional feature dropouts for robustness
    # SVR doesn't like noisy features, so we let optuna drop some feature groups
    use_history = trial.suggest_categorical("use_history", [True, False])
    use_sex_interactions = trial.suggest_categorical("use_sex_interactions", [True, False])
    
    # Filter features based on Optuna suggestion
    cols_to_use = list(features.columns)
    if not use_history:
        cols_to_use = [col for col in cols_to_use if "history" not in col]
    if not use_sex_interactions:
        cols_to_use = [col for col in cols_to_use if not (col.startswith("female_") or col.startswith("male_"))]
        
    X_subset = features[cols_to_use]
    
    categorical = X_subset.select_dtypes(exclude="number").columns.tolist()
    numerical = X_subset.select_dtypes(include="number").columns.tolist()
    
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("weight", FunctionTransformer(multiply_block, kw_args={"weight": num_weight}, validate=False))
        ]), numerical),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ("weight", FunctionTransformer(multiply_block, kw_args={"weight": cat_weight}, validate=False))
        ]), categorical),
    ])
    
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("svr", SVR(kernel="rbf", C=c, gamma=gamma, epsilon=epsilon, cache_size=1500))
    ])
    
    oof = np.zeros(len(X_subset))
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_idx, valid_idx in folds.split(X_subset):
        x_tr, x_va = X_subset.iloc[train_idx], X_subset.iloc[valid_idx]
        y_tr = target[train_idx]
        
        model.fit(x_tr, y_tr)
        pred = np.clip(model.predict(x_va), 0, 1)
        oof[valid_idx] = pred
        
    mae = mean_absolute_error(target, oof)
    return mae

def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    
    # Generate the richest feature set the user created
    raw = train.drop(columns=[ID_COLUMN, TARGET])
    features = build_v10_winner(raw)
    target = train[TARGET].to_numpy(float)
    
    study = optuna.create_study(direction="minimize")
    print("Starting Optuna search for SVR optimal feature weights and hyperparameters...")
    study.optimize(lambda trial: objective(trial, features, target), n_trials=50, show_progress_bar=True)
    
    print("\n================ Optuna Optimization Finished ================")
    print(f"Best MAE: {study.best_value:.6f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
        
    # Save results
    results_df = study.trials_dataframe()
    results_df = results_df.sort_values("value")
    results_df.to_csv(RESULT_DIR / "svr_optuna_search_results.csv", index=False)
    
if __name__ == "__main__":
    main()
