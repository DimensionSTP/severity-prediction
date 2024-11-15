from typing import Dict, Any, List

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class SeverityDataset:
    def __init__(
        self,
        mode: str,
        data_path: str,
        dataset_name: str,
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
        self.label_column_name = label_column_name
        self.scale_type = scale_type
        if self.scale_type not in ["unscale", "standard", "min-max"]:
            raise ValueError(
                f"Invalid scale execution type: {self.scale_type}. Choose in ['unscale', 'standard', 'min-max']."
            )

    def __call__(self) -> Tuple[pd.DataFrame, pd.Series]:
        dataset = self.load_dataset()
        dataset = self.preprocess_certain_features(dataset)
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

    def preprocess_certain_features(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        remove_chars = str.maketrans("", "", "'\"{}[]:,")
        dataset.columns = [column.translate(remove_chars) for column in dataset.columns]
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
        _, categorical_columns = self.get_columns_by_types(dataset)
        for column in categorical_columns:
            label_encoder = LabelEncoder()
            dataset[column] = label_encoder.fit_transform(dataset[column].astype(str))
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
            data = dataset
            label = 0

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