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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
from sklearn.linear_model import Ridge
import warnings
from pandas.api.types import is_numeric_dtype
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1️⃣  경로 및 모듈 설정
# ------------------------------------------------------------
sys.path.append(str(Path("/Users/bitsaem/Desktop/stress_project/8/5")))
from experiment_team_agephys_selection_v38 import make_features
from experiment_team_feature_transfer_v35 import load_team_functions

DATA_DIR = Path("/Users/bitsaem/Desktop/stress_project/open")
RESULT_DIR = Path("/Users/bitsaem/Desktop/stress_project/8_6")
TARGET = "stress_score"
ID_COLUMN = "ID"

# ------------------------------------------------------------
# 2️⃣  도메인 특화 파생 변수 (가능한 경우에만)   
# ------------------------------------------------------------
AGE_BLOCK = {"age_bmi", "age_map", "age_glucose", "age_bone_density"}
GLUCOSE_BONE = {"age_glucose", "age_bone_density"}

def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """데이터프레임에 도메인 특화 피처를 추가한다.
    - BMI : weight / (height/100)^2
    - Age Group : 20s, 30s, 40s, 50s, 60s
    - TyG index  : log( glucose * triglycerides / 2 )
    - Atherogenic index : (cholesterol - HDL) / HDL
    - LDL/HDL ratio : LDL / HDL
    - Log transforms for skewed variables
    """
    x = df.copy()
    
    # BMI
    if "weight" in x.columns and "height" in x.columns:
        x["BMI"] = x["weight"] / ((x["height"] / 100) ** 2)
    
    # Age Group
    if "age" in x.columns:
        x["age_group"] = (x["age"] // 10 * 10).astype(str)
        
    # TyG index
    if "glucose" in x.columns and "triglycerides" in x.columns:
        tg = x["triglycerides"].replace(0, np.nan)
        x["TyG_index"] = np.log((x["glucose"] * tg) / 2.0 + 1e-6)
        
    # Atherogenic index
    if "cholesterol" in x.columns and "HDL" in x.columns:
        hdl = x["HDL"].replace(0, np.nan)
        x["Atherogenic_index"] = (x["cholesterol"] - hdl) / hdl
        
    # LDL/HDL ratio
    if "LDL" in x.columns and "HDL" in x.columns:
        hdl = x["HDL"].replace(0, np.nan)
        x["LDL_HDL_ratio"] = x["LDL"] / hdl

    # Log transforms for highly skewed variables
    for col in ["triglycerides", "glucose"]:
        if col in x.columns:
            x[f"log_{col}"] = np.log1p(x[col])
            
    # Interaction terms
    if "BMI" in x.columns and "glucose" in x.columns:
        x["BMI_glucose"] = x["BMI"] * x["glucose"]
        
    return x

# ------------------------------------------------------------
# 3️⃣  가중치 함수 (수치/범주 가중치)
# ------------------------------------------------------------
def multiply_block(X, weight):
    return X * weight

# ------------------------------------------------------------
# 4️⃣  기본 모델 예측 (Residual SVR)   
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 5️⃣  메타 피처 결합 함수
# ------------------------------------------------------------
def build_meta_features(base_pred_df: pd.DataFrame) -> pd.DataFrame:
    """base_pred_df 에는 각 베이스 모델의 예측값 컬럼이 들어 있다.
    여기서는 그대로 메타 모델 입력으로 사용한다.
    """
    return base_pred_df.copy()

# ------------------------------------------------------------
# 6️⃣  메인 파이프라인
# ------------------------------------------------------------
def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    add_features, tree_predict, pair_predict, pair_weight = load_team_functions()

    # ---- 데이터 로드 ------------------------------------------------
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    train_raw = train.drop(columns=[ID_COLUMN, TARGET])
    test_raw = test.drop(columns=[ID_COLUMN])
    y = train[TARGET].reset_index(drop=True)
    test_ids = test[ID_COLUMN]

    # ---- 기본 피처 + 추가 도메인 피처 -------------------------------
    train_raw = add_extra_features(train_raw)
    test_raw = add_extra_features(test_raw)

    # 기존 age/physiology 피처 만들기
    train_age_x = make_features(train_raw, add_features, AGE_BLOCK)
    train_gb_x = make_features(train_raw, add_features, GLUCOSE_BONE)
    test_age_x = make_features(test_raw, add_features, AGE_BLOCK)
    test_gb_x = make_features(test_raw, add_features, GLUCOSE_BONE)

    # ---- 베이스 모델 리스트 ------------------------------------------
    # 1) Residual SVR (이전 파이프라인)
    # 2) Gradient Boosting Regressor
    # 3) Random Forest Regressor
    base_models = {
        "svr": {
            "type": "svr",
            "params": {
                "C": 0.06028699440816751,
                "gamma": 0.01819126938986479,
                "epsilon": 0.0012891892933940274,
                "alpha": 0.10604972737467394,
                "cat_weight": 3.6684255472865646,
                "num_weight": 9.926669921990287,
                "use_poly": True,
            },
        },
        "gbr": {
            "type": "gbr",
            "params": {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 4,
                "subsample": 0.8,
            },
        },
        "rf": {
            "type": "rf",
            "params": {
                "n_estimators": 400,
                "max_depth": None,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
            },
        },
        "lgbm": {
            "type": "lgbm",
            "params": {},
        },
        "catboost": {
            "type": "catboost",
            "params": {},
        },
    }

    # ------------------------------------------------------------
    # 7️⃣  OOF 예측 생성 (5‑fold)
    # ------------------------------------------------------------
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = {name: np.zeros(len(train_raw)) for name in base_models}
    test_preds = {name: np.zeros(len(test_raw)) for name in base_models}

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_raw)):
        # ----- Residual SVR -----
        age_pred, gb_pred = v38_components(
            train_age_x, train_gb_x, y, train_idx, valid_idx,
            tree_predict, pair_predict, pair_weight,
        )
        base_pred = np.round(0.5 * age_pred + 0.5 * gb_pred, 2)
        # meta‑SVR 모델 (use_poly=True as per plan)
        cat_w = base_models["svr"]["params"]["cat_weight"]
        num_w = base_models["svr"]["params"]["num_weight"]
        categorical_cols = [c for c in train_age_x.columns if not is_numeric_dtype(train_age_x[c])]
        numeric_cols = [c for c in train_age_x.columns if is_numeric_dtype(train_age_x[c])]
        preproc = ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("weight", FunctionTransformer(multiply_block, kw_args={"weight": num_w}, validate=False)),
            ]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ("weight", FunctionTransformer(multiply_block, kw_args={"weight": cat_w}, validate=False)),
            ]), categorical_cols),
        ])
        svr = SVR(kernel="rbf",
                  C=base_models["svr"]["params"]["C"],
                  gamma=base_models["svr"]["params"]["gamma"],
                  epsilon=base_models["svr"]["params"]["epsilon"],
                  cache_size=500)
        pipe = Pipeline([("preproc", preproc), ("svr", svr)])
        # Train on OOF training split (inner)
        pipe.fit(train_age_x.iloc[train_idx], y.iloc[train_idx])
        # Predict for validation
        oof_preds["svr"][valid_idx] = np.clip(pipe.predict(train_age_x.iloc[valid_idx]), 0, 1)

        # ----- Gradient Boosting -----
        gbr_pipe = Pipeline([
            ("preproc", preproc),
            ("gbr", GradientBoostingRegressor(**base_models["gbr"]["params"], random_state=42)),
        ])
        gbr_pipe.fit(train_age_x.iloc[train_idx], y.iloc[train_idx])
        oof_preds["gbr"][valid_idx] = gbr_pipe.predict(train_age_x.iloc[valid_idx])

        # ----- Random Forest -----
        rf_pipe = Pipeline([
            ("preproc", preproc),
            ("rf", RandomForestRegressor(**base_models["rf"]["params"], random_state=42, n_jobs=-1)),
        ])
        rf_pipe.fit(train_age_x.iloc[train_idx], y.iloc[train_idx])
        oof_preds["rf"][valid_idx] = rf_pipe.predict(train_age_x.iloc[valid_idx])

        # LightGBM
        lgbm_pipe = Pipeline([
            ("preproc", preproc),
            ("lgbm", LGBMRegressor(**base_models["lgbm"]["params"], random_state=42, n_jobs=-1)),
        ])
        lgbm_pipe.fit(train_age_x.iloc[train_idx], y.iloc[train_idx])
        oof_preds["lgbm"][valid_idx] = lgbm_pipe.predict(train_age_x.iloc[valid_idx])

        # CatBoost
        cat_pipe = Pipeline([
            ("preproc", preproc),
            ("cat", CatBoostRegressor(**base_models["catboost"]["params"], random_state=42, verbose=False)),
        ])
        cat_pipe.fit(train_age_x.iloc[train_idx], y.iloc[train_idx])
        oof_preds["catboost"][valid_idx] = cat_pipe.predict(train_age_x.iloc[valid_idx])

        # ---- Test set 예측 (각 모델) -----------------------------------
        test_preds["svr"] += pipe.predict(test_age_x) / n_splits
        test_preds["gbr"] += gbr_pipe.predict(test_age_x) / n_splits
        test_preds["rf"] += rf_pipe.predict(test_age_x) / n_splits
        test_preds["lgbm"] += lgbm_pipe.predict(test_age_x) / n_splits
        test_preds["catboost"] += cat_pipe.predict(test_age_x) / n_splits

    # ------------------------------------------------------------
    # 8️⃣  메타 모델 (Ridge) 학습
    # ------------------------------------------------------------
    meta_train = pd.DataFrame({
        "svr": oof_preds["svr"],
        "gbr": oof_preds["gbr"],
        "rf": oof_preds["rf"],
        "lgbm": oof_preds["lgbm"],
        "catboost": oof_preds["catboost"],
    })
    meta_test = pd.DataFrame({
        "svr": test_preds["svr"],
        "gbr": test_preds["gbr"],
        "rf": test_preds["rf"],
        "lgbm": test_preds["lgbm"],
        "catboost": test_preds["catboost"],
    })
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(meta_train, y)
    final_pred = ridge.predict(meta_test)
    final_pred = np.clip(final_pred, 0, 1)

    # ------------------------------------------------------------
    # 9️⃣  검증 MAE 출력 (전체 OOF)
    # ------------------------------------------------------------
    overall_oof = np.mean([oof_preds[m] for m in base_models], axis=0)
    mae_oof = mean_absolute_error(y, overall_oof)
    print(f"Overall OOF MAE (simple avg of base models): {mae_oof:.6f}")
    mae_meta = mean_absolute_error(y, ridge.predict(meta_train))
    print(f"Meta‑Ridge MAE on OOF meta‑features: {mae_meta:.6f}")

    # ------------------------------------------------------------
    # 10️⃣  제출 파일 저장
    # ------------------------------------------------------------
    submission = pd.DataFrame({ID_COLUMN: test_ids, TARGET: final_pred})
    out_path = RESULT_DIR / "submission_optuna_v47.csv"
    submission.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path.resolve()}")

if __name__ == "__main__":
    main()
