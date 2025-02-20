from typing import Dict, Any, Tuple
import os
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import xgboost as xgb

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner


class XGBTuner:
    def __init__(
        self,
        hparams: Dict[str, Any],
        data: pd.DataFrame,
        label: pd.Series,
        direction: str,
        seed: int,
        num_trials: int,
        objective_name: str,
        metric_name: str,
        early_stop: int,
        num_folds: int,
        hparams_save_path: str,
    ) -> None:
        self.hparams = hparams

        self.data = data
        self.label = label

        self.direction = direction
        self.seed = seed
        self.num_trials = num_trials

        self.objective_name = objective_name
        self.metric_name = metric_name
        self.early_stop = early_stop
        self.num_folds = num_folds
        self.hparams_save_path = hparams_save_path

    def __call__(self) -> None:
        study = optuna.create_study(
            direction=self.direction,
            sampler=TPESampler(seed=self.seed),
            pruner=HyperbandPruner(),
        )
        study.optimize(
            self.optuna_objective,
            n_trials=self.num_trials,
        )
        trial = study.best_trial
        best_score = trial.value
        best_params = trial.params
        print(f"Best score: {best_score}")
        print(f"Parameters: {best_params}")

        os.makedirs(
            self.hparams_save_path,
            exist_ok=True,
        )

        with open(f"{self.hparams_save_path}/best_params.json", "w") as json_file:
            json.dump(
                best_params,
                json_file,
            )

    def eval_metric(
        self,
        pred: np.ndarray,
        dataset: xgb.DMatrix,
    ) -> Tuple[str, float]:
        label = dataset.get_label()
        binary_pred = (pred > 0.5).astype(int)
        metric = f1_score(
            y_true=label,
            y_pred=binary_pred,
        )
        return (
            self.metric_name,
            metric,
        )

    def optuna_objective(
        self,
        trial: optuna.trial.Trial,
    ) -> float:
        params = dict()
        if self.hparams.booster:
            params["booster"] = trial.suggest_categorical(
                name="booster",
                choices=self.hparams.booster,
            )
        params["objective"] = self.objective_name
        params["random_state"] = self.seed
        if self.hparams._lambda:
            params["lambda"] = trial.suggest_loguniform(
                name="lambda",
                low=self.hparams._lambda.low,
                high=self.hparams._lambda.high,
            )
        if self.hparams.alpha:
            params["alpha"] = trial.suggest_loguniform(
                name="alpha",
                low=self.hparams.alpha.low,
                high=self.hparams.alpha.high,
            )
        if self.hparams.max_depth:
            params["max_depth"] = trial.suggest_int(
                name="max_depth",
                low=self.hparams.max_depth.low,
                high=self.hparams.max_depth.high,
                log=self.hparams.max_depth.log,
            )
        if self.hparams.eta:
            params["eta"] = trial.suggest_loguniform(
                name="eta",
                low=self.hparams.eta.low,
                high=self.hparams.eta.high,
            )
        if self.hparams.gamma:
            params["gamma"] = trial.suggest_loguniform(
                name="gamma",
                low=self.hparams.gamma.low,
                high=self.hparams.gamma.high,
            )
        if self.hparams.subsample:
            params["subsample"] = trial.suggest_uniform(
                name="subsample",
                low=self.hparams.subsample.low,
                high=self.hparams.subsample.high,
            )
        if self.hparams.colsample_bytree:
            params["colsample_bytree"] = trial.suggest_uniform(
                name="colsample_bytree",
                low=self.hparams.colsample_bytree.low,
                high=self.hparams.colsample_bytree.high,
            )

        kf = StratifiedKFold(
            n_splits=self.num_folds,
            shuffle=True,
            random_state=self.seed,
        )

        metric_results = []
        for idx in tqdm(kf.split(self.data, self.label)):
            train_data, train_label = self.data.loc[idx[0]], self.label.loc[idx[0]]
            val_data, val_label = self.data.loc[idx[1]], self.label.loc[idx[1]]
            train_dataset = xgb.DMatrix(
                data=train_data,
                label=train_label,
                enable_categorical=True,
            )
            val_dataset = xgb.DMatrix(
                data=val_data,
                label=val_label,
                enable_categorical=True,
            )

            model = xgb.train(
                params=params,
                dtrain=train_dataset,
                evals=[
                    (
                        train_dataset,
                        "train",
                    ),
                    (
                        val_dataset,
                        "validation",
                    ),
                ],
                feval=self.eval_metric,
                maximize=True,
                early_stopping_rounds=self.early_stop,
            )

            metric_result = model.best_score
            metric_results.append(metric_result)
        score = np.mean(metric_results)
        return score
