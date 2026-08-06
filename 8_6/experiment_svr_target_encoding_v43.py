import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("/Users/bitsaem/Desktop/stress_project/open")
RESULT_DIR = Path("/Users/bitsaem/Desktop/stress_project/8/5/8/5_result")
TARGET = "stress_score"
ID_COLUMN = "ID"

def kfold_target_encoding(train_df, cat_cols, target_col, n_splits=5, seed=42):
    encoded_df = train_df.copy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    for col in cat_cols:
        encoded_df[col + '_te'] = np.nan
        
    for train_idx, val_idx in kf.split(train_df):
        X_tr, X_val = train_df.iloc[train_idx], train_df.iloc[val_idx]
        
        for col in cat_cols:
            means = X_tr.groupby(col)[target_col].mean()
            encoded_df.loc[val_idx, col + '_te'] = X_val[col].map(means)
            
    # Fill any remaining NaNs with global mean
    global_mean = train_df[target_col].mean()
    for col in cat_cols:
        encoded_df[col + '_te'] = encoded_df[col + '_te'].fillna(global_mean)
        
    return encoded_df

def feature_engineering(df):
    result = df.copy()
    height_m = result["height"] / 100.0
    result["bmi"] = result["weight"] / (height_m ** 2)
    result["mean_arterial_pressure"] = (result["systolic_blood_pressure"] + 2.0 * result["diastolic_blood_pressure"]) / 3.0
    result["pulse_pressure"] = result["systolic_blood_pressure"] - result["diastolic_blood_pressure"]
    
    # Missing value handling
    cat_cols = ["medical_history", "family_medical_history", "mean_working"]
    for col in cat_cols:
        if col in result.columns:
            result[col] = result[col].fillna("missing")
    return result

def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    
    # Simple imputation for mean_working if it's numeric
    if train_raw['mean_working'].dtype == float or train_raw['mean_working'].dtype == int:
        train_raw['mean_working'] = train_raw['mean_working'].fillna(train_raw['mean_working'].median())
    
    train_fe = feature_engineering(train_raw)
    
    cat_cols = [
        "gender", "activity", "smoke_status", "medical_history", 
        "family_medical_history", "sleep_pattern", "edu_level"
    ]
    
    # 1. Target Encoding
    train_encoded = kfold_target_encoding(train_fe, cat_cols, TARGET, n_splits=5, seed=42)
    
    # Drop original cat cols
    train_encoded = train_encoded.drop(columns=cat_cols)
    train_encoded = train_encoded.drop(columns=[ID_COLUMN])
    
    features = train_encoded.drop(columns=[TARGET])
    target = train_encoded[TARGET].values
    
    # 2. Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # 3. K-Fold Evaluation for SVR
    oof = np.zeros(len(scaled_features))
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for tr_idx, va_idx in cv.split(scaled_features):
        x_tr, x_va = scaled_features[tr_idx], scaled_features[va_idx]
        y_tr, y_va = target[tr_idx], target[va_idx]
        
        # SVR with standard RBF
        model = SVR(kernel='rbf', C=5.0, gamma=0.05, epsilon=0.0)
        model.fit(x_tr, y_tr)
        
        pred = model.predict(x_va)
        oof[va_idx] = np.clip(pred, 0, 1)
        
    mae = mean_absolute_error(target, oof)
    print(f"SVR with Target Encoding MAE: {mae:.6f}")

if __name__ == "__main__":
    main()
