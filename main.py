import dotenv

dotenv.load_dotenv(
    override=True,
)

import os
import warnings

os.environ["HYDRA_FULL_ERROR"] = "1"
warnings.filterwarnings("ignore")

import json

import hydra
from omegaconf import DictConfig

from src.pipelines.pipeline import train, test, predict, tune


@hydra.main(
    config_path="configs/",
    config_name="lgbm.yaml",
)
def main(
    config: DictConfig,
) -> None:
    if config.mode == "train":
        return train(config)
    elif config.mode == "test":
        return test(config)
    elif config.mode == "predict":
        return predict(config)
    elif config.mode == "tune":
        return tune(config)
    else:
        raise ValueError(f"Invalid execution mode: {config.mode}")


if __name__ == "__main__":
    main()
