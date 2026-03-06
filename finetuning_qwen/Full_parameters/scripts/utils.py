import math
import torch

def get_preprocess_function(tokenizer, max_length):
    def process_data(example):
        try:
            human = example["conversations"][0]["value"]
            gpt = example["conversations"][1]["value"]
        except (KeyError, IndexError):
            return {"input_ids": [], "attention_mask": [], "labels": []}

        user_message = [{"role": "user", "content": human}]
        full_messages = [
            {"role": "user", "content": human},
            {"role": "assistant", "content": gpt}
        ]

        # 질문 부분 템플릿 적용 (길이 계산용)
        user_prompt = tokenizer.apply_chat_template(user_message, tokenize=True, add_generation_prompt=True)
        user_prompt_len = len(user_prompt)

        # 전체 프롬프트 토크나이징
        full_prompt = tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_dict=True 
        )
        
        input_ids = full_prompt["input_ids"]
        attention_mask = full_prompt["attention_mask"]

        # 모델이 생성해야 할 부분(assistant)만 Loss를 계산하도록 라벨 마스킹
        labels = list(input_ids)
        labels[:min(user_prompt_len, len(labels))] = [-100] * min(user_prompt_len, len(labels))

        labels = [
            -100 if token == tokenizer.pad_token_id else token for token in labels
        ]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        
    return process_data


def evaluate_model(model, eval_dataloader, accelerator):
    """
    모델의 Evaluation을 수행하고 평균 Loss와 Perplexity를 반환합니다.
    """
    model.eval()
    losses = []
    
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
        
        loss = outputs.loss
        # 다중 GPU(DDP) 환경일 경우 모든 GPU의 Loss를 수집하여 정확한 평균 계산
        loss = accelerator.gather_for_metrics(loss)
        losses.append(loss.mean().item())
        
    eval_loss = sum(losses) / len(losses)
    
    # Loss가 너무 높을 경우 오버플로우 방지
    try:
        eval_ppl = math.exp(eval_loss)
    except OverflowError:
        eval_ppl = float("inf")
        
    return eval_loss, eval_ppl