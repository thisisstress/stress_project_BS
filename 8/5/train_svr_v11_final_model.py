from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from experiment_svr_demographic_interactions_v6 import make_model
from experiment_svr_v10_kernel_tuning_v11 import build_v10_winner


PROJECT_DIR = Path("/Users/bitsaem/Desktop/stress_project")
DATA_DIR = PROJECT_DIR / "open"
RESULT_DIR = PROJECT_DIR / "8/5/8/5_result"
TARGET = "stress_score"
ID_COLUMN = "ID"
C = 5.0
GAMMA = 0.08
EPSILON = 0.0
VERIFIED_MEAN_CV_MAE = 0.162173
VERIFIED_SEED_MAE = {
    "42": 0.1632624252055463,
    "77": 0.1619455266289465,
    "2026": 0.1613116012321137,
}


def validate_model(features: pd.DataFrame, target: np.ndarray):
    oof_frame = pd.DataFrame(index=np.arange(len(features)))
    seed_scores = {}
    fold_rows = []
    print("\n3개 시드 × 5-Fold 교차검증을 시작합니다.")
    for seed in (42, 77, 2026):
        oof = np.zeros(len(features), dtype=float)
        splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(features), 1):
            fold_model = make_model(features, C, GAMMA)
            fold_model.fit(features.iloc[train_idx], target[train_idx])
            prediction = np.clip(
                fold_model.predict(features.iloc[valid_idx]), 0.0, 1.0
            )
            oof[valid_idx] = prediction
            fold_mae = mean_absolute_error(target[valid_idx], prediction)
            fold_rows.append({"seed": seed, "fold": fold, "MAE": fold_mae})
            print(f"Seed {seed} | Fold {fold} MAE: {fold_mae:.6f}", flush=True)
        seed_mae = mean_absolute_error(target, oof)
        seed_scores[seed] = seed_mae
        oof_frame[f"oof_seed_{seed}"] = oof
        print(f"Seed {seed} 전체 OOF MAE: {seed_mae:.6f}\n", flush=True)
    return seed_scores, pd.DataFrame(fold_rows), oof_frame


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    train_features = build_v10_winner(train.drop(columns=[ID_COLUMN, TARGET]))
    test_features = build_v10_winner(test.drop(columns=[ID_COLUMN]))
    if train_features.columns.tolist() != test_features.columns.tolist():
        raise RuntimeError("Train과 Test의 V11 피처 구성이 일치하지 않습니다.")

    target = train[TARGET].to_numpy(float)

    seed_scores, fold_scores, oof_predictions = validate_model(
        train_features, target
    )
    mean_cv_mae = float(np.mean(list(seed_scores.values())))
    oof_predictions.insert(0, TARGET, target)
    oof_predictions.insert(0, ID_COLUMN, train[ID_COLUMN].to_numpy())
    oof_path = RESULT_DIR / "svr_v11_final_oof.csv"
    fold_path = RESULT_DIR / "svr_v11_final_fold_scores.csv"
    oof_predictions.to_csv(oof_path, index=False)
    fold_scores.to_csv(fold_path, index=False)

    model = make_model(train_features, C, GAMMA)
    model.fit(train_features, target)
    prediction = np.clip(model.predict(test_features), 0.0, 1.0)

    model_path = RESULT_DIR / "svr_v11_final_model.joblib"
    joblib.dump(model, model_path)

    submission = sample.copy()
    submission[ID_COLUMN] = test[ID_COLUMN].to_numpy()
    submission[TARGET] = prediction
    submission_path = RESULT_DIR / "submit_svr_v11_final_raw.csv"
    submission.to_csv(submission_path, index=False)

    metadata = {
        "model": "SVR",
        "kernel": "rbf",
        "C": C,
        "gamma": GAMMA,
        "epsilon": EPSILON,
        "feature_version": "V11 full verified feature set",
        "n_input_features": int(train_features.shape[1]),
        "feature_columns": train_features.columns.tolist(),
        "mean_cv_mae": mean_cv_mae,
        "seed_mae": {str(seed): float(score) for seed, score in seed_scores.items()},
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
        "prediction_mean": float(prediction.mean()),
    }
    metadata_path = RESULT_DIR / "svr_v11_final_model_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 저장한 모델을 다시 불러와 예측 재현성을 확인한다.
    loaded_model = joblib.load(model_path)
    loaded_prediction = np.clip(loaded_model.predict(test_features), 0.0, 1.0)
    max_difference = float(np.max(np.abs(prediction - loaded_prediction)))
    if max_difference > 1e-12:
        raise RuntimeError(f"저장 모델의 재현 예측이 다릅니다: {max_difference}")

    print("\n================ SVR V11 최종 모델 ================")
    print(f"피처 수            : {train_features.shape[1]}")
    print(f"모델 설정          : RBF SVR(C={C}, gamma={GAMMA}, epsilon={EPSILON})")
    print(f"검증 평균 CV MAE   : {mean_cv_mae:.6f}")
    print(
        "시드별 CV MAE     : "
        + ", ".join(f"{seed}={score:.6f}" for seed, score in seed_scores.items())
    )
    print(
        f"예측 범위/평균     : {prediction.min():.6f}~{prediction.max():.6f} / "
        f"{prediction.mean():.6f}"
    )
    print(f"저장 재현 최대 차이: {max_difference:.12f}")
    print(f"모델 파일          : {model_path.resolve()}")
    print(f"메타데이터         : {metadata_path.resolve()}")
    print(f"OOF 예측           : {oof_path.resolve()}")
    print(f"Fold별 점수        : {fold_path.resolve()}")
    print(f"제출 파일          : {submission_path.resolve()}")
    print("====================================================")
    print("\n첫 10개 예측값")
    print(submission.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
