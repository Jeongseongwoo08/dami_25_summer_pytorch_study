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
from omegaconf import OmegaConf

from conf.config import LoRAConfig 
from peft import LoraConfig as peftLoraConfig, get_peft_model
from utils import get_preprocess_function, evaluate_model 

@hydra.main(version_base=None, config_name="lora_config")
def main(cfg: LoRAConfig):     
    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    set_seed(cfg.seed)

    # 1. Accelerate 준비
    accelerator = Accelerator(log_with="wandb")

    accelerator.init_trackers(
        project_name="qwen-finetuning-coastral",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    logger.info(f"Loading Model: {cfg.model_name}")

    # 2. tokenizer, model load 
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    
    config = AutoConfig.from_pretrained(cfg.model_name)
    config.max_position_embeddings = cfg.max_length 
    config.use_cache = False 
    config.tie_word_embeddings = False
    
    if hasattr(config, "sliding_window"):
        config.sliding_window = cfg.max_length 
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    )
    
    # LoRA 적용 시작
    lora_config = peftLoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 데이터 로드 및 전처리
    logger.info("Loading and processing dataset...")
    with accelerator.main_process_first():
        dataset = load_dataset(cfg.dataset_name)
        
        # 데이터셋에 validation split이 없다면 생성 (9:1 비율)
        if "validation" not in dataset:
            logger.info("Validation split이 없어 Train 데이터를 9:1로 분할합니다.")
            dataset = dataset["train"].train_test_split(test_size=0.1, seed=cfg.seed)
            train_dataset = dataset["train"]
            eval_dataset = dataset["test"]
        else:
            train_dataset = dataset["train"]
            eval_dataset = dataset["validation"]

        # utils.py의 팩토리 함수를 이용해 전처리 함수 획득
        process_data_fn = get_preprocess_function(tokenizer, cfg.max_length)

        train_dataset = train_dataset.map(process_data_fn, remove_columns=["conversations"])
        eval_dataset = eval_dataset.map(process_data_fn, remove_columns=["conversations"])
    
    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")

    # 4. 학습 및 평가 준비
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=cfg.batch_size)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=cfg.batch_size)
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [ param for name, param in model.named_parameters() if not any(nd in name for nd in no_decay) and param.requires_grad],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [ param for name, param in model.named_parameters() if any (nd in name for nd in no_decay) and param.requires_grad],
            "weight_decay": 0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)

    num_training_steps = cfg.epoch_count * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # eval_dataloader도 accelerator에 추가로 등록해야 합니다.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # 5. 학습 루프
    logger.info("학습 시작")
    
    total_step = 0
    for epoch in range(cfg.epoch_count):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
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

        # --- 매 Epoch 종료 시 평가(Evaluation) 수행 ---
        logger.info(f"Epoch {epoch} 종료. 모델 평가 중...")
        eval_loss, eval_ppl = evaluate_model(model, eval_dataloader, accelerator)
        
        logger.info(f"Epoch {epoch} 평가 결과 | Eval Loss: {eval_loss:.4f} | Eval PPL: {eval_ppl:.4f}")
        accelerator.log(
            {
                "eval_loss": eval_loss,
                "eval_perplexity": eval_ppl,
                "epoch": epoch
            },
            step=total_step
        )

    logger.info("Training finished. Saving model...")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)

        unwrapped_model.save_pretrained("./qwen_finetuned", save_function=accelerator.save, safe_serialization=False)
        tokenizer.save_pretrained("./qwen_finetuned")

    accelerator.end_training()
    logger.info("All done!")

if __name__ == "__main__":
    main()