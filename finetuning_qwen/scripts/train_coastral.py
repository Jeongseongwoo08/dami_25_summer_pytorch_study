import torch
import logging
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler, AutoConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from accelerate.utils import set_seed
import hydra

from conf.coastral_config import coastralConfig 
import wandb
from omegaconf import OmegaConf

@hydra.main(version_base=None, config_name="coastral_config")
def main(cfg:coastralConfig):     
    # logging 설정
    logging.basicConfig(
        format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt = "%m/%d/%Y %H:%M:%S",
        level = logging.INFO,
    )
    logger = logging.getLogger(__name__)

    set_seed(cfg.seed)

    # 1. Accelerate 준비
    accelerator = Accelerator(log_with="wandb")

    accelerator.init_trackers(
        project_name = "qwen-finetuning-coastral",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    logger.info(f"Loading Model: {cfg.model_name}")

    # 2. tokenizer, model load 
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    ''' 이부분은 gpt가 만든 거라서 확인해봐야함 '''
    config = AutoConfig.from_pretrained(cfg.model_name)
    config.max_position_embeddings = cfg.max_length 
    config.use_cache = False 
    
    # tie_word_embeddings는 config 객체에 직접 설정해 줍니다.
    config.tie_word_embeddings = False
    
    if hasattr(config, "sliding_window"):
        config.sliding_window = cfg.max_length 
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        config=config,           # 모든 설정이 담긴 config 사용
        torch_dtype=torch.bfloat16,
        attn_implementation="eager" 
    )
    ''' 여기까지 '''
    # pad_token이 없다면 eos_token으로 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 데이터 로드 및 전처리
    logger.info("Loading and processing dataset...")
    with accelerator.main_process_first():
        dataset = load_dataset(cfg.dataset_name, split="train")

        def process_data(example):
            try:
                human= example["conversations"][0]["value"]
                gpt= example["conversations"][1]["value"]
            except (KeyError, IndexError):
                return {"input_ids": [], "attention_mask": [], "labels":[]}

            # 질문만 포함된 메세지
            user_message = [{"role":"user", "content":human}]
            # 전체 메세지
            full_messages = [
                {"role": "user", "content":human},
                {"role": "assistant", "content":gpt}
            ]

            # 질문 부분 템플릿 적용 (길이 계산용)
            user_prompt = tokenizer.apply_chat_template(user_message, tokenize=True, add_generation_prompt=True)
            user_prompt_len = len(user_prompt)

            full_prompt = tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
                max_length=cfg.max_length,
                padding="max_length",
                truncation=True,
                return_dict=True 
            )
            
            input_ids = full_prompt["input_ids"]
            attention_mask = full_prompt["attention_mask"]

            labels = list(input_ids)
            # labels 슬라이싱 시 범위를 넘지 않도록 처리
            labels[:min(user_prompt_len, len(labels))] = [-100] * min(user_prompt_len, len(labels))

            labels = [
                -100 if token == tokenizer.pad_token_id else token for token in labels
            ]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }

        tokenized_dataset = dataset.map(process_data, remove_columns=["conversations"])
    
    tokenized_dataset.set_format("torch")

    # 학습 준비
    train_dataloader = DataLoader(tokenized_dataset, shuffle=True, batch_size = cfg.batch_size)
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [ param for name, param in model.named_parameters() if not any(nd in name for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [ param for name, param in model.named_parameters() if any (nd in name for nd in no_decay)],
            "weight_decay": 0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr = cfg.lr)

    num_training_steps = cfg.epoch_count * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear", optimizer = optimizer, num_warmup_steps = 0, num_training_steps = num_training_steps
    )

    # accelerator prepare
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)

    # 학습 루프
    logger.info("학습 시작")
    model.train()

    total_step = 0
    for epoch in range(cfg.epoch_count):
        for step, batch in enumerate(train_dataloader):
            outputs = model(
                input_ids = batch["input_ids"],
                attention_mask = batch["attention_mask"],
                labels = batch["labels"],
            )
            loss = outputs.loss

            accelerator.backward(loss)

            accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                accelerator.log(
                    {
                        "train_loss": loss.item(),
                        "epoch": epoch,
                        "learning_rate": current_lr
                    },
                    step=total_step
                )
                logger.info(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | LR: {current_lr:.6f}")

            total_step += 1

    logger.info("Training finished. Saving model...")
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    unwrapped_model.save_pretrained("./qwen_finetuned", save_function=accelerator.save)
    tokenizer.save_pretrained("./qwen_finetuned")

    accelerator.end_training()
    logger.info("All done!")

if __name__ == "__main__":
    main()