from typing import Dict, Any, List
import os
import json
import joblib

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class SeverityDataset:
    def __init__(
        self,
        mode: str,
        data_path: str,
        dataset_name: str,
        columns_mapping_file_path: str,
        timedelta_features: Dict[str, List[str]],
        is_all_features: bool,
        unusing_features: List[str],
        label_encoder_path: str,
        label_column_name: str,
        scale_type: str,
    ) -> None:
        self.mode = mode
        if self.mode not in ["train", "test", "predict", "tune"]:
            raise ValueError(
                f"Invalid mode: {self.mode}. Choose in ['train', 'test', 'predict', 'tune']."
            )

        self.data_path = data_path
        self.dataset_name = dataset_name
        self.columns_mapping_file_path = columns_mapping_file_path
        self.timedelta_features = timedelta_features
        self.is_all_features = is_all_features
        self.unusing_features = unusing_features
        self.label_encoder_path = label_encoder_path
        self.label_column_name = label_column_name
        self.scale_type = scale_type
        if self.scale_type not in ["unscale", "standard", "min-max"]:
            raise ValueError(
                f"Invalid scale execution type: {self.scale_type}. Choose in ['unscale', 'standard', 'min-max']."
            )

    def __call__(self) -> Dict[str, Any]:
        dataset = self.load_dataset()
        dataset = self.map_columns(dataset)
        dataset = self.select_features(dataset)
        dataset = self.preprocess_certain_features(dataset)
        dataset = self.process_date_columns(dataset)
        dataset = self.interpolate_dataset(dataset)
        dataset = self.get_preprocessed_dataset(dataset)
        data = dataset["data"]
        label = dataset["label"]
        return {
            "data": data,
            "label": label,
        }

    def load_dataset(self) -> pd.DataFrame:
        if self.mode == "train" or self.mode == "test":
            dataset = pd.read_csv(
                f"{self.data_path}/{self.dataset_name}_{self.mode}.csv"
            )
        elif self.mode == "predict":
            dataset = pd.read_csv(f"{self.data_path}/{self.dataset_name}_test.csv")
        else:
            dataset = pd.read_csv(f"{self.data_path}/{self.dataset_name}_train.csv")
        return dataset

    def map_columns(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        try:
            with open(self.columns_mapping_file_path, "r", encoding="utf-8") as f:
                mapping_info = json.load(f)
        except FileNotFoundError:
            raise ValueError(
                f"Mapping file not found: {self.columns_mapping_file_path}"
            )

        columns_mapping = {
            column: mapping_info.get(
                column,
                column,
            )
            for column in dataset.columns
        }
        dataset = dataset.rename(columns=columns_mapping)
        return dataset

    def get_date_columns(
        self,
        dataset: pd.DataFrame,
    ) -> List[str]:
        is_date_column = ~dataset.apply(pd.api.types.is_numeric_dtype) & dataset.apply(
            lambda col: pd.to_datetime(col, errors="coerce").notna().any()
        )
        date_columns = list(dataset.columns[is_date_column])
        return date_columns

    def get_timedelta_features(
        self,
        dataset: pd.DataFrame,
        date_columns: List[str],
    ) -> pd.DataFrame:
        if not date_columns:
            return dataset

        filtered_features = {
            key: columns
            for key, columns in self.timedelta_features.items()
            if all(column in date_columns for column in columns)
        }

        date_features = [
            column for columns in filtered_features.values() for column in columns
        ]
        dataset[date_features] = dataset[date_features].apply(
            pd.to_datetime,
            errors="coerce",
        )

        timedelta_features = {
            key: (dataset[columns[1]] - dataset[columns[0]]).dt.days
            for key, columns in filtered_features.items()
        }

        dataset = pd.concat(
            [
                dataset,
                pd.DataFrame(timedelta_features),
            ],
            axis=1,
        )

        return dataset

    def select_features(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        if self.is_all_features:
            return dataset

        unusing_features = [
            feature for feature in self.unusing_features if feature in dataset.columns
        ]
        dataset = dataset.drop(columns=unusing_features)
        return dataset

    def preprocess_certain_features(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        remove_chars = str.maketrans("", "", "'\"{}[]:,")
        dataset.columns = [column.translate(remove_chars) for column in dataset.columns]
        return dataset

    def process_date_columns(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        date_columns = []
        for column in dataset.columns:
            if pd.api.types.is_numeric_dtype(dataset[column]):
                continue
            try:
                checked_column = pd.to_datetime(
                    dataset[column],
                    errors="coerce",
                )
                if checked_column.notna().any():
                    date_columns.append(column)
            except Exception:
                continue

        if not date_columns:
            return dataset

        for date_column in date_columns:
            dataset[date_column] = pd.to_datetime(
                dataset[date_column],
                errors="coerce",
            )
            dataset[f"{date_column}_year"] = (
                dataset[date_column].dt.year.fillna(0).astype(int)
            )
            dataset[f"{date_column}_month"] = (
                dataset[date_column].dt.month.fillna(0).astype(int)
            )
            dataset[f"{date_column}_day"] = (
                dataset[date_column].dt.day.fillna(0).astype(int)
            )
            dataset = dataset.drop(columns=[date_column])
        return dataset

    def get_columns_by_types(
        self,
        dataset: pd.DataFrame,
    ) -> Dict[str, List[str]]:
        continuous_columns = []
        categorical_columns = []
        for column in dataset.columns:
            if pd.api.types.is_numeric_dtype(dataset[column]):
                continuous_columns.append(column)
            else:
                categorical_columns.append(column)
        return {
            "continuous_columns": continuous_columns,
            "categorical_columns": categorical_columns,
        }

    def interpolate_dataset(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        columns = self.get_columns_by_types(dataset)
        continuous_columns, categorical_columns = (
            columns["continuous_columns"],
            columns["categorical_columns"],
        )
        dataset[continuous_columns] = dataset[continuous_columns].interpolate(
            method="linear",
            axis=0,
        )
        dataset[categorical_columns] = dataset[categorical_columns].fillna("NULL")
        return dataset

    def encode_categorical_features(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        os.makedirs(
            self.label_encoder_path,
            exist_ok=True,
        )

        columns = self.get_columns_by_types(dataset)
        categorical_columns = columns["categorical_columns"]
        for column in categorical_columns:
            label_encoder_file_path = f"{self.label_encoder_path}/{column}_encoder.pkl"
            try:
                label_encoder = joblib.load(label_encoder_file_path)
                dataset[column] = label_encoder.fit_transform(
                    dataset[column].astype(str)
                )
            except:
                label_encoder = LabelEncoder()
                dataset[column] = label_encoder.fit_transform(
                    dataset[column].astype(str)
                )
                joblib.dump(
                    label_encoder,
                    label_encoder_file_path,
                )
        return dataset

    def get_preprocessed_dataset(
        self,
        dataset: pd.DataFrame,
    ) -> Dict[str, Any]:
        if self.mode in ["train", "test", "tune"]:
            data = dataset.drop(
                [self.label_column_name],
                axis=1,
            )
            label = dataset[self.label_column_name]
        else:
            try:
                data = dataset.drop(
                    [self.label_column_name],
                    axis=1,
                )
                label = dataset[self.label_column_name]
            except:
                data = dataset
                dataset[self.label_column_name] = 0
                label = dataset[self.label_column_name]

        if self.scale_type == "standard":
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            data = pd.DataFrame(
                scaled_data,
                columns=data.columns,
            )
        elif self.scale_type == "min-max":
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            data = pd.DataFrame(
                scaled_data,
                columns=data.columns,
            )
        else:
            pass
        return {
            "data": data,
            "label": label,
        }
