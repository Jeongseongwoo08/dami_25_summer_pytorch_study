import hydra
import torch
from accelerate import Accelerator
from bitsandbytes.optim import PagedAdamW8bit
from dotenv import load_dotenv
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Gemma3ForCausalLM,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_pt_utils import get_parameter_names

from dami_25_summer_pytorch_study.professor_code.config.sft_config import SFTConfig
from train.dataset import DataCollatorForYelp, YelpDataset
from dami_25_summer_pytorch_study.professor_code.train.utils import save_model


def evaluate(model: PreTrainedModel, eval_dataloader: DataLoader, accelerator: Accelerator) -> Tuple[float, float]:
    model.eval()
    total_loss = 0
    total_eval_steps = 0

    with torch.inference_mode():
        for input_ids, attention_mask, labels in tqdm(
            eval_dataloader, desc="Evaluating", leave=False, disable=not accelerator.is_main_process, mininterval=1
        ):
            eval_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            eval_loss = eval_outputs.loss
            total_loss += eval_loss.item()
            total_eval_steps += 1

    avg_eval_loss = total_loss / total_eval_steps
    ppl = exp(avg_eval_loss)
    return avg_eval_loss, ppl


@hydra.main(version_base=None, config_name="sft_config")
def main(config: SFTConfig):
    load_dotenv()

    config.output_dir = os.path.join(config.output_dir, config.run_name)
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="wandb" if config.use_wandb else None,
    )

    # Configure logger to only log from main process
    if not accelerator.is_main_process:
        logger.remove()

    logger.info(f"Number of processes: {accelerator.num_processes}")
    if os.environ.get("WANDB_API_KEY"):
        accelerator.init_trackers(
            project_name=config.project_name, config=config.__dict__, init_kwargs={"wandb": {"name": config.run_name}}
        )
        logger.info(f"WandB initialized with project: {config.project_name}, run: {config.run_name}")
    else:
        logger.warning("WANDB_API_KEY not found or not configured in .env file. Disabling WandB logging.")

    # Initialize model
    with accelerator.main_process_first():
        # Initialize tokenizer
        logger.info(f"Loading tokenizer from {config.model_uri}")
        tokenizer = AutoTokenizer.from_pretrained(config.model_uri, use_fast=True, trust_remote_code=True)
        if tokenizer.pad_token is None:
            # Set PAD token as EOS token
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        logger.info(f"Loading model from {config.model_uri}")
        if "gemma" in config.model_uri.lower():
            model: PreTrainedModel = Gemma3ForCausalLM.from_pretrained(
                config.model_uri,
                dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="eager",
                use_cache=False,
            )
        else:
            model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                config.model_uri,
                dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                use_cache=False,
            )

        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

    # Create Optimizer
    logger.info("Initializing Optimizer")
    forbidden_name_patterns = [r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
    decay_parameters = set(get_parameter_names(model, [torch.nn.LayerNorm], forbidden_name_patterns))
    grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": config.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if n not in decay_parameters], "weight_decay": 0.0},
    ]
    optimizer = PagedAdamW8bit(grouped_parameters, lr=config.learning_rate)
    logger.info(f"Using per_device_train_batch_size: {config.per_device_train_batch_size}")
    logger.info(f"Using gradient_accumulation_steps: {config.gradient_accumulation_steps}")
    logger.info(f"Using learning_rate: {config.learning_rate}")
    logger.info(f"Using warmup_steps: {config.warmup_steps}")

    # Prepare dataset
    logger.info("Preparing dataset")
    train_dataset = YelpDataset(data_path=config.data_path, tokenizer=tokenizer, max_length=config.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
        collate_fn=DataCollatorForYelp(tokenizer=tokenizer, mode="appended"),
    )

    eval_dataset = YelpDataset(data_path=config.eval_data_path, tokenizer=tokenizer, max_length=config.max_length)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.per_device_eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
        collate_fn=DataCollatorForYelp(tokenizer=tokenizer, mode="appended"),
    )

    # Create learning rate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps * accelerator.num_processes,
        num_training_steps=len(train_loader) * config.num_train_epochs,
    )

    # Prepare with Accelerate
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, lr_scheduler)
    total_steps = len(train_loader) * config.num_train_epochs // accelerator.num_processes

    if eval_dataloader is not None:
        eval_dataloader = accelerator.prepare(eval_dataloader)

    logger.info("Starting training")
    model.train()
    step = 0

    for epoch in range(1, config.num_train_epochs + 1):
        logger.info(f"Starting epoch {epoch}/{config.num_train_epochs}")

        for input_ids, attention_mask, labels in train_loader:
            with accelerator.accumulate(model):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            step += 1

            # Logging
            if step % config.logging_steps == 0 and accelerator.is_main_process:
                lr = lr_scheduler.get_last_lr()[0]
                loss = loss.item()
                ppl = exp(loss)
                logger.info(f"Epoch {epoch}, Step {step}/{total_steps} Loss: {loss:.4f}, PPL: {ppl:.4f}, LR: {lr:.2e}")
                accelerator.log({"train/loss": loss, "train/epoch": epoch, "train/learning_rate": lr}, step=step)

            # Evaluation
            if eval_dataloader and step % config.eval_steps == 0:
                eval_loss, eval_ppl = evaluate(model, eval_dataloader, accelerator)
                model.train()

                logger.info(
                    f"Epoch {epoch}, Step {step}/{total_steps}, Eval Loss: {eval_loss:.4f}, Eval PPL: {eval_ppl:.4f}"
                )
                accelerator.log({"eval/loss": eval_loss, "eval/ppl": eval_ppl}, step=step)

            # Save checkpoint
            if step % config.save_steps == 0:
                save_model(model, tokenizer, os.path.join(config.output_dir, f"step_{step}"), accelerator)
                logger.info(f"Saving model checkpoint to {os.path.join(config.output_dir, f'step_{step}')}")

    # Final evaluation
    if eval_dataloader:
        final_eval_loss, final_eval_ppl = evaluate(model, eval_dataloader, accelerator)
        logger.info(f"Final Evaluation Loss: {final_eval_loss:.4f}, Final Evaluation PPL: {final_eval_ppl:.4f}")
        accelerator.log(
            {"eval/final_loss": final_eval_loss, "eval/final_ppl": final_eval_ppl, "eval/step": step}, step=step
        )

    # Save final model
    save_model(model, tokenizer, os.path.join(config.output_dir, f"step_{step}"), accelerator)
    logger.info(f"Saving final model to {os.path.join(config.output_dir, f'step_{step}')}")

    logger.info("Training completed!")
    if config.use_wandb:
        accelerator.end_training()


if __name__ == "__main__":
    main()