from typing import Union

from omegaconf import DictConfig
from hydra.utils import instantiate

from ..utils.setup import SetUp

from ..tuners.cb_tuner import CBTuner
from ..tuners.lgbm_tuner import LGBMTuner
from ..tuners.xgb_tuner import XGBTuner


def train(
    config: DictConfig,
) -> None:
    setup = SetUp(config)

    dataset = setup.get_dataset()
    architecture = setup.get_architecture()

    dataset = dataset()
    data, label = dataset["data"], dataset["label"]

    architecture.train(
        data=data,
        label=label,
    )


def test(
    config: DictConfig,
) -> None:
    setup = SetUp(config)

    dataset = setup.get_dataset()
    architecture = setup.get_architecture()

    dataset = dataset()
    data, label = dataset["data"], dataset["label"]

    architecture.test(
        data=data,
        label=label,
    )


def predict(
    config: DictConfig,
) -> None:
    setup = SetUp(config)

    dataset = setup.get_dataset()
    architecture = setup.get_architecture()

    dataset = dataset()
    data = dataset["data"]

    architecture.predict(data=data)


def tune(
    config: DictConfig,
) -> None:
    setup = SetUp(config)

    dataset = setup.get_dataset()

    dataset = dataset()
    data, label = dataset["data"], dataset["label"]

    tuner: Union[LGBMTuner, XGBTuner, CBTuner] = instantiate(
        config.tuner,
        data=data,
        label=label,
    )
    tuner()
