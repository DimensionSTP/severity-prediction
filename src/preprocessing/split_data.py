import dotenv

dotenv.load_dotenv(
    override=True,
)

import os
import json

import pandas as pd
from sklearn.model_selection import train_test_split

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="lgbm.yaml",
)
def split_data(
    config: DictConfig,
) -> None:
    dataset = pd.read_csv(f"{config.connected_dir}/data/{config.dataset_name}.csv")
    dataset = map_columns(
        config=config,
        dataset=dataset,
    )
    train_dataset, test_dataset = train_test_split(
        dataset,
        test_size=config.split_ratio,
        random_state=config.seed,
        shuffle=True,
        stratify=dataset[config.label_column_name],
    )
    train_dataset = train_dataset.reset_index(drop=True)
    test_dataset = test_dataset.reset_index(drop=True)

    train_dataset.to_csv(
        f"{config.connected_dir}/data/{config.dataset_name}_train.csv",
        index=False,
    )
    test_dataset.to_csv(
        f"{config.connected_dir}/data/{config.dataset_name}_test.csv",
        index=False,
    )


def map_columns(
    config: DictConfig,
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    columns_mapping_file_path = os.path.join(
        config.connected_dir,
        config.columns_mapping_file_path,
    )

    try:
        with open(columns_mapping_file_path, "r", encoding="utf-8") as f:
            mapping_info = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Mapping file not found: {columns_mapping_file_path}")

    columns_mapping = {
        column: mapping_info.get(
            column,
            column,
        )
        for column in dataset.columns
    }
    dataset = dataset.rename(columns=columns_mapping)
    return dataset


if __name__ == "__main__":
    split_data()
