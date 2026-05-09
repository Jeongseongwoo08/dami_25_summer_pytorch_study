from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


@dataclass
class SFTConfig:
    project_name: str = "privacy_aware_knowledge_distillation"
    run_name: str = "sft_qwen3-1.7B"

    model_uri: str = "Qwen/Qwen3-1.7B"
    data_path: str = "resources/data/yelp_review/train.tsv"
    eval_data_path: str = "resources/data/yelp_review/test.tsv"

    max_length: int = 512
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.001
    warmup_steps: int = 200

    logging_steps: int = 20
    eval_steps: int = 100
    save_steps: int = 2000
    num_workers: int = 2
    output_dir: str = "./checkpoints/sft"
    use_wandb: bool = False


cs = ConfigStore.instance()
cs.store(name="sft_config", node=SFTConfig)