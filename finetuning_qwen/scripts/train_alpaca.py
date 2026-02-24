import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator

# 1. Accelerate 준비
accelerator = Accelerator()

# 2. 모델 및 토크나이저 로드
model_name = "Qwen/Qwen3-1.7B" # Qwen3가 아직 공개 전이라면 최신 버전 사용
# (사용하시는 Qwen3-1.7B가 있다면 그 이름을 쓰시면 됩니다)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, # 메모리 절약 및 속도 향상
    # device_map="auto" -> Accelerate가 자동으로 장치를 관리하므로 여기서는 뺍니다.
)

# Qwen은 pad_token이 없을 수 있어 eos_token으로 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# 3. 데이터셋 로드 (예: yahma/alpaca-cleaned - 가볍고 좋은 지시 데이터셋)
dataset = load_dataset("yahma/alpaca-cleaned", split="train") # 실습용 500개만

# 4. 전처리 함수 (Qwen 템플릿 맞추기)
def preprocess_function(examples):
    inputs = []
    
    # 데이터셋의 instruction, input, output을 꺼내서 포맷팅
    for instruction, input_text, output in zip(examples['instruction'], examples['input'], examples['output']):
        # 프롬프트 구성 (User 입력)
        if input_text:
            user_content = f"{instruction}\nInput: {input_text}"
        else:
            user_content = instruction
            
        # Qwen 채팅 구조 생성
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output} # 학습할 정답
        ]
        
        # apply_chat_template으로 텍스트 변환 (토큰화는 아직)
        # tokenize=False로 하여 텍스트 전체 구조를 먼저 봅니다.
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs.append(text)
    
    # 토큰화 수행
    model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    
    # Label 생성 (Causal LM은 input 자체가 정답지)
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    
    # 패딩 토큰 부분은 loss 계산에서 제외 (-100으로 설정)
    model_inputs["labels"][model_inputs["input_ids"] == tokenizer.pad_token_id] = -100
    
    return model_inputs

# 데이터셋에 전처리 적용
tokenized_datasets = dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=dataset.column_names
)
tokenized_datasets.set_format("torch")


# 5. 데이터 로더 준비
train_dataloader = DataLoader(tokenized_datasets, shuffle=True, batch_size=2) # 메모리에 맞게 조절

# 6. 옵티마이저 설정
optimizer = AdamW(model.parameters(), lr=2e-5)

# 7. 스케줄러 설정
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# ★★★ 8. Accelerate의 prepare로 모든 객체 감싸기 (가장 중요!) ★★★
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)

# 9. 학습 시작
model.train()
accelerator.print("학습 시작!") # print 대신 accelerator.print 사용 권장

for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataloader):
        # Accelerate가 device 관리를 해주므로 .to(device) 불필요
        
        # 순전파 (Forward)
        outputs = model(**batch)
        loss = outputs.loss
        
        # 역전파 (Backward) -> accelerator.backward 사용!
        accelerator.backward(loss)
        
        # 파라미터 업데이트
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        # --- 중간 저장 로직 추가 ---
        if (step + 1) % 1000 == 0:
            accelerator.wait_for_everyone()
            checkpoint_dir = f"./qwen_checkpoints/step_{step+1}"
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                checkpoint_dir, 
                save_function=accelerator.save,
                is_main_process=accelerator.is_main_process
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(checkpoint_dir)
            accelerator.print(f"Step {step+1} 체크포인트 저장 완료!")
        
        if step % 10 == 0:
            accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item()}")

# 10. 모델 저장
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained("./qwen_finetuned", save_function=accelerator.save)
tokenizer.save_pretrained("./qwen_finetuned")

accelerator.print("학습 완료 및 저장됨.")