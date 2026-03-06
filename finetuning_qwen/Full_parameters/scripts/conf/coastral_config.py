from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

@dataclass
class coastralConfig:
    model_name: str = "Qwen/Qwen3-1.7B"
    dataset_name: str = "coastral/korean-writing-style-instruct"
    epoch_count : int = 3
    lr: float = 0.0001
    batch_size: int = 4
    max_length: int = 1024
    seed: int = 42
    weight_decay: float = 0.01

cs = ConfigStore.instance()
cs.store(name="coastral_config", node=coastralConfig)