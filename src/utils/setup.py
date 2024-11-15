from typing import Union

from omegaconf import DictConfig
from hydra.utils import instantiate

from ..datasets.severity_dataset import SeverityDataset

from ..architectures.lgbm_architecture import LGBMArchitecture
from ..architectures.xgb_architecture import XGBArchitecture
from ..architectures.cb_architecture import CBArchitecture


class SetUp:
    def __init__(self, config: DictConfig):
        self.config = config

    def get_dataset(self) -> SeverityDataset:
        dataset: SeverityDataset = instantiate(self.config.dataset)
        return dataset

    def get_architecture(
        self,
    ) -> Union[LGBMArchitecture, XGBArchitecture, CBArchitecture]:
        architecture: Union[LGBMArchitecture, XGBArchitecture, CBArchitecture] = (
            instantiate(self.config.architecture)
        )
        return architecture
