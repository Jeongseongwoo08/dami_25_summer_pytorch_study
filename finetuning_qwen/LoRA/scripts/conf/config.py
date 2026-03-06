from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

@dataclass
class BaseConfig:
    model_name: str = "Qwen/Qwen3-1.7B"
    dataset_name: str = "coastral/korean-writing-style-instruct"
    epoch_count : int = 3
    lr: float = 0.0001
    batch_size: int = 4
    max_length: int = 1024
    seed: int = 42
    weight_decay: float = 0.01

@dataclass
class LoRAConfig(BaseConfig):
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

cs = ConfigStore.instance()
cs.store(
    name="coastral_config", 
    node=BaseConfig
)
cs.store(
    name="lora_config",
    node=LoRAConfig
)