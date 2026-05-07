import torch
import logging
from unsloth import FastLanguageModel
from datasets import load_dataset
from accelerate.utils import set_seed
import hydra
import os

from trl import SFTTrainer, SFTConfig

from conf.config import MainConfig 
import utils

@hydra.main(version_base=None, config_name="qwen_sft_config")
def main(cfg: MainConfig):
    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    set_seed(cfg.base.seed)

    os.environ["WANDB_PROJECT"] = "qwen-finetuning-coastral_sft"

    logger.info(f"Loading Model: {cfg.base.model_name}")

    # 2. tokenizer, model load 
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = cfg.base.model_name,
        max_seq_length = cfg.base.max_length,
        load_in_4bit = True,
        dtype = torch.bfloat16,
        trust_remote_code = True
        # atten_implementation은 unsloth가 자동으로 설정해줌
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    
    
    # 3. 데이터 로드 및 전처리
    logger.info("Loading and processing dataset...")
    dataset = load_dataset(cfg.base.dataset_name)
        
        # 데이터셋에 validation split이 없다면 생성 (9:1 비율)
    if "validation" not in dataset:
        logger.info("Validation split이 없어 Train 데이터를 9:1로 분할합니다.")
        dataset = dataset["train"].train_test_split(test_size=0.1, seed=cfg.base.seed)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    else:
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]

    # utils.py의 팩토리 함수를 이용해 전처리 함수 획득
    process_data_fn = utils.get_preprocess_function(tokenizer, cfg.base.max_length)

    train_dataset = train_dataset.map(process_data_fn, remove_columns=["conversations"])
    eval_dataset = eval_dataset.map(process_data_fn, remove_columns=["conversations"])
    
    # LoRA 적용
    # lora_config = peftLoraConfig(
    #     r=cfg.lora.lora_r,
    #     lora_alpha=cfg.lora.lora_alpha,
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    #     lora_dropout=cfg.lora.lora_dropout,
    #     bias="none",
    #     task_type="CAUSAL_LM"
    # )

    model = FastLanguageModel.get_peft_model(
        model,
        r = cfg.lora.lora_r,
        lora_alpha = cfg.lora.lora_alpha,
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
            lora_dropout = cfg.lora.lora_dropout,
            bias = "none",
            use_gradient_checkpointing="unsloth"
    )

    sft_config = SFTConfig(
        output_dir="./qwen_finetuned_sfttrainer",
        learning_rate=cfg.sft.lr,
        per_device_train_batch_size=cfg.sft.batch_size,
        per_device_eval_batch_size=cfg.sft.batch_size,
        num_train_epochs=cfg.sft.epoch_count,
        weight_decay=cfg.sft.weight_decay,
        max_length=cfg.base.max_length, # max_seq_length -> max_length
        bf16=True,
        eval_strategy="epoch", # 매 epoch 끝마다 평가
        save_strategy="epoch", #매 epoch 끝마다 모델 저장
        report_to="wandb",
        # save_safetensors=False # 저장 포맷 에러 방지
    )

    model.config.use_cache = False # gradient checkpoint 충돌 방지

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        args=sft_config,
    )

    logger.info("학습 시작 (TRL SFTTrainer)")
    trainer.train()

    # 모델 저장
    trainer.save_model("./qwen_finetuned_sfttrainer")
    logger.info("save complete")

    FastLanguageModel.for_inference(model) #inference 최적화
    

if __name__ == "__main__":
    main()
