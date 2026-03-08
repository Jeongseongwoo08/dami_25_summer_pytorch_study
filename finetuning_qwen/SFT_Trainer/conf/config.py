from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

@dataclass
class BaseConfig:
    model_name: str = "Qwen/Qwen3-1.7B"
    dataset_name: str = "coastral/korean-writing-style-instruct"
    max_length: int = 1024
    seed: int = 42

@dataclass
class LoRAParams:
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

@dataclass
class SFTParams:
    lr: float = 0.0001
    epoch_count: int = 3
    batch_size: int = 4
    weight_decay: float = 0.01

# 메인 설정에서 세 가지를 하나로 묶음
@dataclass
class MainConfig:
    base: BaseConfig = field(default_factory=BaseConfig)
    lora: LoRAParams = field(default_factory=LoRAParams)
    sft: SFTParams = field(default_factory=SFTParams)

cs = ConfigStore.instance()
cs.store(
    name="qwen_sft_config",
    node=MainConfig
)