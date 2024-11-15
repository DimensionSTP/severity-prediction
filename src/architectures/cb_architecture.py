import os
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

import catboost as cb

import wandb
from wandb.integration.catboost import WandbCallback

import matplotlib.pyplot as plt


class CBArchitecture:
    def __init__(
        self,
        objective_name: str,
        metric_name: str,
        project_name: str,
        user_name: str,
        save_detail: str,
        model_save_path: str,
        result_summary_path: str,
    ) -> None:
        self.objective_name = objective_name
        self.metric_name = metric_name

        self.project_name = project_name
        self.user_name = user_name
        self.save_detail = save_detail

        self.model_save_path = model_save_path
        self.result_summary_path = result_summary_path

    def train(
        self,
        data: pd.DataFrame,
        label: pd.Series,
        num_folds: int,
        seed: int,
        is_tuned: str,
        hparams_save_path: str,
        plt_save_path: str,
    ) -> None:
        cat_features = [
            column for column in data.columns if data[column].dtype == "object"
        ]

        kf = StratifiedKFold(
            n_splits=num_folds,
            shuffle=True,
            random_state=seed,
        )
        if is_tuned == "tuned":
            params = json.load(
                open(
                    f"{hparams_save_path}/best_params.json",
                    "rt",
                    encoding="UTF-8",
                )
            )
            params["verbose"] = -1
        elif is_tuned == "untuned":
            params = {
                "boosting_type": "Plain",
                "objective": self.objective_name,
                "metric": self.metric_name,
                "random_seed": seed,
            }
        else:
            raise ValueError(f"Invalid is_tuned argument: {is_tuned}")

        wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=self.run_name,
        )

        model = cb.CatBoostRegressor(**params)

        metric_results = []
        for i, idx in enumerate(tqdm(kf.split(data, label))):
            train_data, train_label = data.loc[idx[0]], label.loc[idx[0]]
            val_data, val_label = data.loc[idx[1]], label.loc[idx[1]]

            model.fit(
                train_data,
                train_label,
                cat_features=cat_features,
                callbacks=[WandbCallback()],
            )

            os.makedirs(
                self.model_save_path,
                exist_ok=True,
            )
            model.save_model(f"{self.model_save_path}/fold{i}.txt")

            pred = model.predict(val_data)
            metric_result = np.sqrt(
                mean_squared_error(
                    val_label,
                    pred,
                )
            )
            metric_results.append(metric_result)
        avg_metric_result = np.mean(metric_results)
        print(f"average {self.metric_name}: {avg_metric_result}")

        result = {
            "model_type": "CatBoost",
            "used_features": data.columns.tolist(),
            "num_folds": num_folds,
            self.metric_name: avg_metric_result,
        }
        result_df = pd.DataFrame.from_dict(
            result,
            orient="index",
        ).T

        save_path = os.path.join(
            self.result_summary_path,
            "train",
        )
        os.makedirs(
            save_path,
            exist_ok=True,
        )

        result_file = f"{save_path}/result_summary.csv"
        if os.path.isfile(result_file):
            original_result_df = pd.read_csv(result_file)
            new_result_df = pd.concat(
                [
                    original_result_df,
                    result_df,
                ],
                ignore_index=True,
            )
            new_result_df.to_csv(
                result_file,
                encoding="utf-8-sig",
                index=False,
            )
        else:
            result_df.to_csv(
                result_file,
                encoding="utf-8-sig",
                index=False,
            )

        feature_importance = model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        plt.barh(
            range(len(sorted_idx)),
            feature_importance[sorted_idx],
            align="center",
        )
        plt.yticks(
            range(len(sorted_idx)),
            data.columns[sorted_idx],
        )
        plt.title("Feature Importance")

        os.makedirs(
            plt_save_path,
            exist_ok=True,
        )

        plt.savefig(
            f"{plt_save_path}/num_folds={num_folds}-metric_result={avg_metric_result}.png"
        )

    def test(
        self,
        data: pd.DataFrame,
        label: pd.Series,
    ) -> None:
        model_files = os.listdir(self.model_save_path)
        metric_results = []
        for model_file in tqdm(model_files):
            model = cb.CatBoostRegressor()
            model.load_model(f"{self.model_save_path}/{model_file}")
            pred = model.predict(data) / len((model_files))
            metric_result = np.sqrt(
                mean_squared_error(
                    label,
                    pred,
                )
            )
            metric_results.append(metric_result)
        avg_metric_result = np.mean(metric_results)
        print(f"average {self.metric_name}: {avg_metric_result}")

        result = {
            "model_type": "CatBoost",
            "used_features": data.columns.tolist(),
            "num_models": len(model_files),
            self.metric_name: avg_metric_result,
        }
        result_df = pd.DataFrame.from_dict(
            result,
            orient="index",
        ).T

        save_path = os.path.join(
            self.result_summary_path,
            "test",
        )
        os.makedirs(
            save_path,
            exist_ok=True,
        )

        result_file = f"{save_path}/result_summary.csv"
        if os.path.isfile(result_file):
            original_result_df = pd.read_csv(result_file)
            new_result_df = pd.concat(
                [
                    original_result_df,
                    result_df,
                ],
                ignore_index=True,
            )
            new_result_df.to_csv(
                result_file,
                encoding="utf-8-sig",
                index=False,
            )
        else:
            result_df.to_csv(
                result_file,
                encoding="utf-8-sig",
                index=False,
            )

    def predict(
        self,
        data: pd.DataFrame,
        submission_save_path: str,
        submission_save_name: str,
    ) -> None:
        model_files = os.listdir(self.model_save_path)
        pred_mean = np.zeros((len(data),))
        for model_file in tqdm(model_files):
            model = cb.CatBoostRegressor()
            model.load_model(f"{self.model_save_path}/{model_file}")
            pred = model.predict(data) / len((model_files))
            pred_mean += pred

        submission = pd.DataFrame(
            pred_mean.astype(int),
            columns=["target"],
        )

        os.makedirs(
            submission_save_path,
            exist_ok=True,
        )

        submission.to_csv(
            f"{submission_save_path}/{submission_save_name}.csv",
            index=False,
        )
