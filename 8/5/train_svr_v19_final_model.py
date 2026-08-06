from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from experiment_svr_block_weighting_v17 import make_block_weighted_model
from experiment_svr_v10_kernel_tuning_v11 import build_v10_winner


PROJECT_DIR = Path("/Users/bitsaem/Desktop/stress_project")
DATA_DIR = PROJECT_DIR / "open"
RESULT_DIR = PROJECT_DIR / "8/5/8/5_result"
TARGET = "stress_score"
ID_COLUMN = "ID"
SEEDS = (42, 77, 2026)
C = 5.0
GAMMA = 0.035
EPSILON = 0.0
CATEGORICAL_WEIGHT = 5.0


def validate(features, target):
    oof = pd.DataFrame(index=np.arange(len(features)))
    fold_rows = []
    seed_scores = {}
    print("3개 시드 × 5-Fold 교차검증을 시작합니다.")
    for seed in SEEDS:
        seed_oof = np.zeros(len(features), dtype=float)
        splitter = KFold(5, shuffle=True, random_state=seed)
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(features), 1):
            model = make_block_weighted_model(
                features, C, GAMMA, CATEGORICAL_WEIGHT
            )
            model.fit(features.iloc[train_idx], target[train_idx])
            pred = np.clip(model.predict(features.iloc[valid_idx]), 0.0, 1.0)
            seed_oof[valid_idx] = pred
            fold_mae = mean_absolute_error(target[valid_idx], pred)
            fold_rows.append({"seed": seed, "fold": fold, "MAE": fold_mae})
            print(f"Seed {seed} | Fold {fold} MAE: {fold_mae:.6f}", flush=True)
        score = mean_absolute_error(target, seed_oof)
        seed_scores[seed] = score
        oof[f"oof_seed_{seed}"] = seed_oof
        print(f"Seed {seed} OOF MAE: {score:.6f}\n", flush=True)
    oof["oof_mean"] = oof.mean(axis=1)
    return seed_scores, oof, pd.DataFrame(fold_rows)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    train_features = build_v10_winner(train.drop(columns=[ID_COLUMN, TARGET]))
    test_features = build_v10_winner(test.drop(columns=[ID_COLUMN]))
    if train_features.columns.tolist() != test_features.columns.tolist():
        raise RuntimeError("Train과 Test의 피처 구성이 일치하지 않습니다.")
    target = train[TARGET].to_numpy(float)

    seed_scores, oof_predictions, fold_scores = validate(train_features, target)
    mean_seed_mae = float(np.mean(list(seed_scores.values())))
    ensemble_oof_mae = mean_absolute_error(target, oof_predictions["oof_mean"])
    rounded_oof_mae = mean_absolute_error(
        target, np.round(oof_predictions["oof_mean"].to_numpy(), 2)
    )

    oof_output = oof_predictions.copy()
    oof_output.insert(0, TARGET, target)
    oof_output.insert(0, ID_COLUMN, train[ID_COLUMN].to_numpy())
    oof_path = RESULT_DIR / "svr_v19_final_oof.csv"
    fold_path = RESULT_DIR / "svr_v19_final_fold_scores.csv"
    oof_output.to_csv(oof_path, index=False)
    fold_scores.to_csv(fold_path, index=False)

    final_model = make_block_weighted_model(
        train_features, C, GAMMA, CATEGORICAL_WEIGHT
    )
    final_model.fit(train_features, target)
    prediction = np.clip(final_model.predict(test_features), 0.0, 1.0)

    model_path = RESULT_DIR / "svr_v19_final_model.joblib"
    joblib.dump(final_model, model_path)
    loaded = joblib.load(model_path)
    reproduced = np.clip(loaded.predict(test_features), 0.0, 1.0)
    max_difference = float(np.max(np.abs(prediction - reproduced)))
    if max_difference > 1e-12:
        raise RuntimeError(f"저장 모델 예측이 재현되지 않습니다: {max_difference}")

    raw_submission = sample.copy()
    raw_submission[ID_COLUMN] = test[ID_COLUMN].to_numpy()
    raw_submission[TARGET] = prediction
    raw_path = RESULT_DIR / "submit_svr_v19_final_raw.csv"
    raw_submission.to_csv(raw_path, index=False)

    rounded_submission = raw_submission.copy()
    rounded_submission[TARGET] = np.round(prediction, 2)
    rounded_path = RESULT_DIR / "submit_svr_v19_final_round2.csv"
    rounded_submission.to_csv(rounded_path, index=False)

    metadata = {
        "model": "SVR",
        "kernel": "rbf",
        "C": C,
        "gamma": GAMMA,
        "epsilon": EPSILON,
        "categorical_weight": CATEGORICAL_WEIGHT,
        "feature_version": "V11 full 78 features",
        "n_features": int(train_features.shape[1]),
        "feature_columns": train_features.columns.tolist(),
        "seed_MAE": {str(seed): float(value) for seed, value in seed_scores.items()},
        "mean_seed_MAE": mean_seed_mae,
        "ensemble_oof_MAE": float(ensemble_oof_mae),
        "rounded_ensemble_oof_MAE": float(rounded_oof_mae),
    }
    metadata_path = RESULT_DIR / "svr_v19_final_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n================ SVR V19 최종 모델 ================")
    print(f"피처 수             : {train_features.shape[1]}")
    print(
        f"설정                : C={C}, gamma={GAMMA}, epsilon={EPSILON}, "
        f"cat_weight={CATEGORICAL_WEIGHT}"
    )
    print(f"시드 평균 CV MAE    : {mean_seed_mae:.6f}")
    print(f"3시드 OOF 평균 MAE  : {ensemble_oof_mae:.6f}")
    print(f"OOF 평균 Round2 MAE : {rounded_oof_mae:.6f}")
    print(
        "시드별 MAE          : "
        + ", ".join(f"{seed}={score:.6f}" for seed, score in seed_scores.items())
    )
    print(f"저장 재현 최대 차이 : {max_difference:.12f}")
    print(f"모델                : {model_path.resolve()}")
    print(f"OOF                 : {oof_path.resolve()}")
    print(f"Fold 점수           : {fold_path.resolve()}")
    print(f"Raw 제출            : {raw_path.resolve()}")
    print(f"Round2 제출         : {rounded_path.resolve()}")
    print(f"메타데이터          : {metadata_path.resolve()}")
    print("====================================================")
    print("\nRaw 첫 10개 예측값")
    print(raw_submission.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
