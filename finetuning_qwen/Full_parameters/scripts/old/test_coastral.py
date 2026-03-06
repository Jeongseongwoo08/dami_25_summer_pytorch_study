import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_long_response():
    model_path = "./qwen_finetuned"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    test_questions = [
        "오늘 날씨가 너무 좋은데 산책하기 좋은 장소 추천해줄래?",
        "취업 준비 때문에 너무 스트레스 받는데 조언 좀 해줘.",
        "라면 맛있게 끓이는 비법을 옆집 언니처럼 친절하게 설명해줘.",
        "부장님께 보낼 정중한 휴가 신청 메일 초안을 작성해줘."
    ]

    print("\n🚀 [학습된 모델의 전체 답변 출력]")
    print("=" * 80)

    for q in test_questions:
        messages = [{"role": "user", "content": q}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs, 
                max_new_tokens=400, # 답변 길이를 충분히 늘렸습니다.
                do_sample=True, 
                temperature=0.7,
                repetition_penalty=1.1 # 같은 말 반복 방지
            )
        
        full_text = tokenizer.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        
        # <think> 태그 내부 내용 삭제 후 본론만 추출
        cleaned_text = re.sub(r'<think>.*?</think>', '', full_text, flags=re.DOTALL).strip()

        print(f"Q: {q}")
        print(f"A: {cleaned_text}") # [:60]을 제거하여 전체 문장이 나옵니다!
        print("-" * 80)

if __name__ == "__main__":
    get_long_response()