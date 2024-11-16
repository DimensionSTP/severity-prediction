import os
from typing import Dict, Any, List
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold

import catboost as cb

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner


class CBTuner:
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

    @property
    def cat_features(self) -> List[str]:
        return [
            column
            for column in self.data.columns
            if self.data[column].dtype == "object"
        ]

    def optuna_objective(
        self,
        trial: optuna.trial.Trial,
    ) -> float:
        params = dict()
        params["booster"] = "Plain"
        params["loss_function"] = self.objective_name
        params["eval_metric"] = self.metric_name
        params["random_seed"] = self.seed
        if self.hparams.iterations:
            params["iterations"] = trial.suggest_int(
                name="iterations",
                low=self.hparams.iterations.low,
                high=self.hparams.iterations.high,
                step=self.hparams.iterations.step,
                log=self.hparams.iterations.log,
            )
        if self.hparams.learning_rate:
            params["learning_rate"] = trial.suggest_float(
                name="learning_rate",
                low=self.hparams.learning_rate.low,
                high=self.hparams.learning_rate.high,
                log=self.hparams.learning_rate.log,
            )
        if self.hparams.depth:
            params["depth"] = trial.suggest_int(
                name="depth",
                low=self.hparams.depth.low,
                high=self.hparams.depth.high,
                log=self.hparams.depth.log,
            )
        if self.hparams.l2_leaf_reg:
            params["l2_leaf_reg"] = trial.suggest_float(
                name="l2_leaf_reg",
                low=self.hparams.l2_leaf_reg.low,
                high=self.hparams.l2_leaf_reg.high,
                log=self.hparams.l2_leaf_reg.log,
            )
        if self.hparams.model_size_reg:
            params["model_size_reg"] = trial.suggest_float(
                name="model_size_reg",
                low=self.hparams.model_size_reg.low,
                high=self.hparams.model_size_reg.high,
                log=self.hparams.model_size_reg.log,
            )
        if self.hparams.rsm:
            params["rsm"] = trial.suggest_float(
                name="rsm",
                low=self.hparams.rsm.low,
                high=self.hparams.rsm.high,
                log=self.hparams.rsm.log,
            )
        if self.hparams.subsample:
            params["subsample"] = trial.suggest_float(
                name="subsample",
                low=self.hparams.subsample.low,
                high=self.hparams.subsample.high,
                log=self.hparams.subsample.log,
            )
        if self.hparams.border_count:
            params["border_count"] = trial.suggest_int(
                name="border_count",
                low=self.hparams.border_count.low,
                high=self.hparams.border_count.high,
                log=self.hparams.border_count.log,
            )
        if self.hparams.feature_border_type:
            params["feature_border_type"] = trial.suggest_categorical(
                name="feature_border_type",
                choices=self.hparams.feature_border_type,
            )
        if self.hparams.bootstrap_type:
            params["bootstrap_type"] = trial.suggest_categorical(
                name="bootstrap_type",
                choices=self.hparams.bootstrap_type,
            )
        if self.hparams.grow_policy:
            params["grow_policy"] = trial.suggest_categorical(
                name="grow_policy",
                choices=self.hparams.grow_policy,
            )
        if self.hparams.leaf_estimation_method:
            params["leaf_estimation_method"] = trial.suggest_categorical(
                name="leaf_estimation_method",
                choices=self.hparams.leaf_estimation_method,
            )
        if self.hparams.random_strength:
            params["random_strength"] = trial.suggest_int(
                name="random_strength",
                low=self.hparams.random_strength.low,
                high=self.hparams.random_strength.high,
                log=self.hparams.random_strength.log,
            )
        if self.hparams.bagging_temperature:
            params["bagging_temperature"] = trial.suggest_float(
                name="bagging_temperature",
                low=self.hparams.bagging_temperature.low,
                high=self.hparams.bagging_temperature.high,
                log=self.hparams.bagging_temperature.log,
            )

        if params["bootstrap_type"] == "Bayesian":
            del params["subsample"]
        else:
            del params["bagging_temperature"]

        kf = StratifiedKFold(
            n_splits=self.num_folds,
            shuffle=True,
            random_state=self.seed,
        )

        model = cb.CatBoostClassifier(**params)

        cat_features = [
            column
            for column in self.data.columns
            if self.data[column].dtype == "object"
        ]

        metric_results = []
        for idx in tqdm(kf.split(self.data, self.label)):
            train_data, train_label = self.data.loc[idx[0]], self.label.loc[idx[0]]
            val_data, val_label = self.data.loc[idx[1]], self.label.loc[idx[1]]
            train_dataset = cb.Pool(
                data=train_data,
                label=train_label,
                cat_features=cat_features,
            )
            val_dataset = cb.Pool(
                data=val_data,
                label=val_label,
                cat_features=cat_features,
            )

            model.fit(
                X=train_dataset,
                eval_set=[
                    train_dataset,
                    val_dataset,
                ],
                use_best_model=True,
                plot=False,
                early_stopping_rounds=self.early_stop,
            )

            metric_result = model.best_score_["validation"][self.metric_name]
            metric_results.append(metric_result)
        score = np.mean(metric_results)
        return score
