import os
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold

import catboost as cb

import wandb
from wandb.integration.catboost import WandbCallback

import matplotlib.pyplot as plt


class CBArchitecture:
    def __init__(
        self,
        project_name: str,
        user_name: str,
        save_detail: str,
        num_folds: int,
        seed: int,
        is_tuned: bool,
        hparams_save_path: str,
        objective_name: str,
        metric_name: str,
        early_stop: int,
        model_save_path: str,
        result_summary_path: str,
        plt_save_path: str,
        label_column_name: str,
        submission_save_path: str,
    ) -> None:
        self.project_name = project_name
        self.user_name = user_name
        self.save_detail = save_detail

        self.num_folds = num_folds
        self.seed = seed
        self.is_tuned = is_tuned
        self.hparams_save_path = hparams_save_path

        self.objective_name = objective_name
        self.metric_name = metric_name
        self.early_stop = early_stop

        self.model_save_path = model_save_path
        self.result_summary_path = result_summary_path
        self.plt_save_path = plt_save_path

        self.label_column_name = label_column_name
        self.submission_save_path = submission_save_path

    def train(
        self,
        data: pd.DataFrame,
        label: pd.Series,
    ) -> None:
        wandb.init(
            project=self.project_name,
            entity=self.user_name,
            name=self.save_detail,
        )

        kf = StratifiedKFold(
            n_splits=self.num_folds,
            shuffle=True,
            random_state=self.seed,
        )
        if self.is_tuned == "tuned":
            params = json.load(
                open(
                    f"{self.hparams_save_path}/best_params.json",
                    "rt",
                    encoding="UTF-8",
                )
            )
            params["loss_function"] = self.objective_name
            params["eval_metric"] = self.metric_name
            params["random_seed"] = self.seed
            params["verbose"] = False

            if params["bootstrap_type"] == "Bayesian":
                del params["subsample"]
            else:
                del params["bagging_temperature"]
        elif self.is_tuned == "untuned":
            params = {
                "loss_function": self.objective_name,
                "eval_metric": self.metric_name,
                "random_seed": self.seed,
                "verbose": False,
            }
        else:
            raise ValueError(f"Invalid is_tuned argument: {self.is_tuned}")

        model = cb.CatBoostClassifier(**params)

        cat_features = [
            column for column in data.columns if data[column].dtype == "object"
        ]

        metric_results = []
        for i, idx in enumerate(tqdm(kf.split(data, label))):
            train_data, train_label = data.loc[idx[0]], label.loc[idx[0]]
            val_data, val_label = data.loc[idx[1]], label.loc[idx[1]]
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
                callbacks=[WandbCallback()],
            )

            os.makedirs(
                self.model_save_path,
                exist_ok=True,
            )
            model.save_model(fname=f"{self.model_save_path}/fold{i}.txt")

            metric_result = model.best_score_["validation_1"][self.metric_name]
            metric_results.append(metric_result)
        avg_metric_result = np.mean(metric_results)
        print(f"average {self.metric_name}: {avg_metric_result}")

        result = {
            "model_type": "CatBoost",
            "used_features": data.columns.tolist(),
            "num_folds": self.num_folds,
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
            self.plt_save_path,
            exist_ok=True,
        )

        plt.savefig(f"{self.plt_save_path}/metric_result={avg_metric_result}.png")

    def test(
        self,
        data: pd.DataFrame,
        label: pd.Series,
    ) -> None:
        wandb.init(
            project=self.project_name,
            entity=self.user_name,
            name=self.save_detail,
        )

        test_dataset = cb.Pool(
            data=data,
            label=label,
        )

        model = cb.CatBoostClassifier()

        model_files = os.listdir(self.model_save_path)
        metric_results = []
        for model_file in tqdm(model_files):
            model.load_model(f"{self.model_save_path}/{model_file}")
            metric_result = model.eval_metrics(
                data=test_dataset,
                metrics=[self.metric_name],
            )[self.metric_name][-1]
            metric_results.append(metric_result)

            wandb.log(
                {
                    "model_file": model_file.split(".")[0],
                    self.metric_name: metric_result,
                }
            )

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
    ) -> None:
        predict_dataset = cb.Pool(data=data)

        model = cb.CatBoostClassifier()

        model_files = os.listdir(self.model_save_path)
        pred_mean = np.zeros((len(data),))
        for model_file in tqdm(model_files):
            model.load_model(f"{self.model_save_path}/{model_file}")
            pred = model.predict(data=predict_dataset) / len((model_files))
            pred_mean += pred

        submission = pd.DataFrame(
            np.round(pred_mean).astype(int),
            columns=[self.label_column_name],
        )

        os.makedirs(
            self.submission_save_path,
            exist_ok=True,
        )

        submission.to_csv(
            f"{self.submission_save_path}/prediction.csv",
            index=False,
        )
