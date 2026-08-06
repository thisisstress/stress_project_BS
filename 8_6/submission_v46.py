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
import warnings

warnings.filterwarnings("ignore")

sys.path.append(str(Path("/Users/bitsaem/Desktop/stress_project/8/5")))
from experiment_team_agephys_selection_v38 import make_features
from experiment_team_feature_transfer_v35 import load_team_functions

DATA_DIR = Path("/Users/bitsaem/Desktop/stress_project/open")
RESULT_DIR = Path("/Users/bitsaem/Desktop/stress_project/8_6")
TARGET = "stress_score"
ID_COLUMN = "ID"

AGE = {"age_bmi", "age_map", "age_glucose", "age_bone_density"}
GLUCOSE_BONE = {"age_glucose", "age_bone_density"}

def multiply_block(X, weight):
    return X * weight

def component_prediction(train_x, valid_x, train_y, tree_predict, pair_predict, pair_weight):
    tree = tree_predict(train_x, valid_x, train_y)
    pair = pair_predict(train_x, valid_x, train_y)
    return np.clip(np.round((1 - pair_weight) * tree + pair_weight * pair, 2), 0, 1)

def v38_components(age_x, gb_x, y, train_idx, valid_idx, tree_predict, pair_predict, pair_weight):
    # Base models naturally handle separation of train and valid
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
    result["age_pred"] = age_pred
    result["gb_pred"] = gb_pred
    result["v38_prediction"] = np.round(0.5 * age_pred + 0.5 * gb_pred, 2)
    result["age_gb_disagreement"] = np.abs(age_pred - gb_pred)
    return result

def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    add_features, tree_predict, pair_predict, pair_weight = load_team_functions()
    
    # Data Load
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    
    train_raw = train.drop(columns=[ID_COLUMN, TARGET])
    y = train[TARGET].reset_index(drop=True)
    test_raw = test.drop(columns=[ID_COLUMN])
    test_ids = test[ID_COLUMN]
    
    print("Making base features... (Independent row-wise ops)")
    train_age_x = make_features(train_raw, add_features, AGE)
    train_gb_x = make_features(train_raw, add_features, GLUCOSE_BONE)
    
    test_age_x = make_features(test_raw, add_features, AGE)
    test_gb_x = make_features(test_raw, add_features, GLUCOSE_BONE)
    
    # 1. Generate OOF Predictions for Train set to safely train Meta-Model
    print("Generating OOF predictions for Meta-Model training...")
    oof_age = np.zeros(len(train_raw))
    oof_gb = np.zeros(len(train_raw))
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, valid_idx in kf.split(train_raw):
        age_pred, gb_pred = v38_components(
            train_age_x, train_gb_x, y, train_idx, valid_idx,
            tree_predict, pair_predict, pair_weight
        )
        oof_age[valid_idx] = age_pred
        oof_gb[valid_idx] = gb_pred
        
    train_base_pred = np.round(0.5 * oof_age + 0.5 * oof_gb, 2)
    meta_train = add_meta_columns(train_age_x, oof_age, oof_gb)
    residual_y = y.to_numpy() - train_base_pred
    
    # 2. Generate Final Base Predictions for Test set (Train on FULL Train set)
    print("Training base models on FULL train set for Test predictions...")
    full_train_idx = np.arange(len(train_raw))
    full_test_idx = np.arange(len(test_raw))
    
    test_age, test_gb = v38_components(
        pd.concat([train_age_x, test_age_x]).reset_index(drop=True),
        pd.concat([train_gb_x, test_gb_x]).reset_index(drop=True),
        y, full_train_idx, len(train_raw) + full_test_idx,
        tree_predict, pair_predict, pair_weight
    )
    
    test_base_pred = np.round(0.5 * test_age + 0.5 * test_gb, 2)
    meta_test = add_meta_columns(test_age_x, test_age, test_gb)
    
    # 3. Train Meta-SVR and predict Test
    print("Training Meta SVR and generating final submission...")
    # Best Params from Optuna task
    c = 0.06028699440816751
    gamma = 0.01819126938986479
    epsilon = 0.0012891892933940274
    alpha = 0.10604972737467394
    cat_weight = 3.6684255472865646
    num_weight = 9.926669921990287
    
    categorical = meta_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numerical = [column for column in meta_train.columns if column not in categorical]
    
    # STRICT DATA LEAKAGE PREVENTION: pipeline fits ONLY on meta_train
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
    
    model.fit(meta_train, residual_y)
    
    # SVR predicts residual on Test
    test_correction = model.predict(meta_test)
    final_test_pred = np.clip(test_base_pred + alpha * test_correction, 0, 1)
    
    # 4. Save Submission
    submission = pd.DataFrame({
        ID_COLUMN: test_ids,
        TARGET: final_test_pred
    })
    
    out_path = RESULT_DIR / "submission_optuna_v46.csv"
    submission.to_csv(out_path, index=False)
    print(f"Submission saved successfully to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
