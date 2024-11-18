from typing import Dict, Any, Tuple
import os
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner


class LGBMTuner:
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

    def optuna_objective(
        self,
        trial: optuna.trial.Trial,
    ) -> float:
        params = dict()
        params["boosting_type"] = "gbdt"
        params["objective"] = self.objective_name
        params["metric"] = self.metric_name
        params["seed"] = self.seed
        params["verbosity"] = -1
        if self.hparams.learning_rate:
            params["learning_rate"] = trial.suggest_float(
                name="learning_rate",
                low=self.hparams.learning_rate.low,
                high=self.hparams.learning_rate.high,
                log=self.hparams.learning_rate.log,
            )
        if self.hparams.n_estimators:
            params["n_estimators"] = trial.suggest_int(
                name="n_estimators",
                low=self.hparams.n_estimators.low,
                high=self.hparams.n_estimators.high,
                log=self.hparams.n_estimators.log,
            )
        if self.hparams.lambda_l1:
            params["lambda_l1"] = trial.suggest_loguniform(
                name="lambda_l1",
                low=self.hparams.lambda_l1.low,
                high=self.hparams.lambda_l1.high,
            )
        if self.hparams.lambda_l2:
            params["lambda_l2"] = trial.suggest_loguniform(
                name="lambda_l2",
                low=self.hparams.lambda_l2.low,
                high=self.hparams.lambda_l2.high,
            )
        if self.hparams.num_leaves:
            params["num_leaves"] = trial.suggest_int(
                name="num_leaves",
                low=self.hparams.num_leaves.low,
                high=self.hparams.num_leaves.high,
                log=self.hparams.num_leaves.log,
            )
        if self.hparams.max_depth:
            params["max_depth"] = trial.suggest_int(
                name="max_depth",
                low=self.hparams.max_depth.low,
                high=self.hparams.max_depth.high,
                log=self.hparams.max_depth.log,
            )
        if self.hparams.feature_fraction:
            params["feature_fraction"] = trial.suggest_uniform(
                name="feature_fraction",
                low=self.hparams.feature_fraction.low,
                high=self.hparams.feature_fraction.high,
            )
        if self.hparams.bagging_fraction:
            params["bagging_fraction"] = trial.suggest_uniform(
                name="bagging_fraction",
                low=self.hparams.bagging_fraction.low,
                high=self.hparams.bagging_fraction.high,
            )
        if self.hparams.bagging_freq:
            params["bagging_freq"] = trial.suggest_int(
                name="bagging_freq",
                low=self.hparams.bagging_freq.low,
                high=self.hparams.bagging_freq.high,
                log=self.hparams.bagging_freq.log,
            )
        if self.hparams.min_child_samples:
            params["min_child_samples"] = trial.suggest_int(
                name="min_child_samples",
                low=self.hparams.min_child_samples.low,
                high=self.hparams.min_child_samples.high,
                log=self.hparams.min_child_samples.log,
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
        if self.hparams.reg_alpha:
            params["reg_alpha"] = trial.suggest_uniform(
                name="reg_alpha",
                low=self.hparams.reg_alpha.low,
                high=self.hparams.reg_alpha.high,
            )
        if self.hparams.reg_lambda:
            params["reg_lambda"] = trial.suggest_uniform(
                name="reg_lambda",
                low=self.hparams.reg_lambda.low,
                high=self.hparams.reg_lambda.high,
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
            train_dataset = lgb.Dataset(
                data=train_data,
                label=train_label,
            )
            val_dataset = lgb.Dataset(
                data=val_data,
                label=val_label,
            )

            model = lgb.train(
                params=params,
                train_set=train_dataset,
                valid_sets=[
                    train_dataset,
                    val_dataset,
                ],
                valid_names=("validation"),
                callbacks=[lgb.early_stopping(stopping_rounds=self.early_stop)],
            )

            metric_result = model.best_score["validation"][params["metric"]]
            metric_results.append(metric_result)
        score = np.mean(metric_results)
        return score
