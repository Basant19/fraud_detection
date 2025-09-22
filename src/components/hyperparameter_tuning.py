# src/components/hyperparameter_tuning.py

import os
import sys
import pickle
import json
import optuna
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, fbeta_score
from dotenv import load_dotenv

from src.entity.config import HyperparameterTuningConfig
from src.entity.artifacts import DataTransformationArtifacts, HyperparameterTuningArtifacts
from src.exception import CustomException
from src.logger import logging

# Candidate models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class HyperparameterTuner:
    def __init__(self, config: HyperparameterTuningConfig):
        self.config = config
        load_dotenv()
        self.f2_scorer = make_scorer(fbeta_score, beta=2)

    def initiate_hyperparameter_tuning(
        self, transformation_artifact: DataTransformationArtifacts
    ) -> HyperparameterTuningArtifacts:
        try:
            logging.info("Starting hyperparameter tuning...")

            # ---------------- Load Transformed Data ----------------
            train_data = np.load(transformation_artifact.transformed_train_path, allow_pickle=True)
            X_train, y_train = train_data["X_train"], train_data["y_train"]

            # ---------------- Define Models & Search Space ----------------
            search_spaces = {
                "RandomForest": {
                    "model": RandomForestClassifier(random_state=self.config.random_state, n_jobs=-1),
                    "params": {
                        "n_estimators": [100, 200, 500],
                        "max_depth": [None, 10, 20, 30],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4]
                    }
                },
                "XGBoost": {
                    "model": XGBClassifier(
                        random_state=self.config.random_state,
                        use_label_encoder=False,
                        eval_metric="logloss"
                    ),
                    "params": {
                        "n_estimators": [100, 200, 500],
                        "max_depth": [3, 5, 7, 10],
                        "learning_rate": [0.01, 0.05, 0.1, 0.2],
                        "subsample": [0.6, 0.8, 1.0],
                        "colsample_bytree": [0.6, 0.8, 1.0]
                    }
                }
            }

            best_model = None
            best_params = None
            best_score = -1

            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.random_state)

            for model_name, entry in search_spaces.items():
                model = entry["model"]
                param_grid = entry["params"]

                logging.info(f"Tuning {model_name} using {self.config.search_strategy}")

                if self.config.search_strategy == "grid":
                    search = GridSearchCV(
                        model,
                        param_grid,
                        scoring=self.f2_scorer,
                        cv=cv,
                        n_jobs=-1,
                        verbose=1
                    )
                elif self.config.search_strategy == "random":
                    search = RandomizedSearchCV(
                        model,
                        param_distributions=param_grid,
                        n_iter=10,
                        scoring=self.f2_scorer,
                        cv=cv,
                        n_jobs=-1,
                        verbose=1,
                        random_state=self.config.random_state
                    )
                elif self.config.search_strategy == "optuna":
                    def objective(trial):
                        params = {k: trial.suggest_categorical(k, v) for k, v in param_grid.items()}
                        model.set_params(**params)
                        scores = []
                        for train_idx, val_idx in cv.split(X_train, y_train):
                            model.fit(X_train[train_idx], y_train[train_idx])
                            preds = model.predict(X_train[val_idx])
                            scores.append(fbeta_score(y_train[val_idx], preds, beta=2))
                        return np.mean(scores)

                    study = optuna.create_study(direction="maximize")
                    study.optimize(objective, n_trials=20)
                    best_params = study.best_params
                    best_score = study.best_value
                    model.set_params(**best_params)
                    model.fit(X_train, y_train)
                    best_model = model
                    continue
                else:
                    raise ValueError(f"Unsupported search strategy: {self.config.search_strategy}")

                search.fit(X_train, y_train)
                if search.best_score_ > best_score:
                    best_score = search.best_score_
                    best_model = search.best_estimator_
                    best_params = search.best_params_

            # ---------------- Save Best Model & Params ----------------
            os.makedirs(os.path.dirname(self.config.tuned_model_path), exist_ok=True)

            with open(self.config.tuned_model_path, "wb") as f:
                pickle.dump(best_model, f)

            with open(self.config.best_params_path, "w") as f:
                json.dump(best_params, f, indent=4)

            logging.info(f"Best Model Tuned with score={best_score:.4f}")

            return HyperparameterTuningArtifacts(
                tuned_model_path=self.config.tuned_model_path,
                best_params_path=self.config.best_params_path,
                best_score=best_score
            )

        except Exception as e:
            logging.error("Error in hyperparameter tuning", exc_info=True)
            raise CustomException(e, sys)
