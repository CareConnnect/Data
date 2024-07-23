import os
import torch
from transformers import GPTJForCausalLM, AutoTokenizer
import pandas as pd
from datasets import Dataset

os.environ['HUGGINGFACE_API_KEY'] = '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'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# MODEL_AND_TOKENIZER
model_name = 'EleutherAI/gpt-j-6B'
tokenizer = AutoTokenizer.from_pretrained(model_name, token = os.environ['HUGGINGFACE_API_KEY'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GPTJForCausalLM.from_pretrained(model_name, token = os.environ['HUGGINGFACE_API_KEY']).to(device)
tokenizer.pad_token = tokenizer.eos_token

def create_prompts(df, num_per_disease):
    prompts = []
    for idx, row in df.iterrows():
        for _ in range(num_per_disease):
            prompt = f"""
            Generate a conversational dataset entry for common elderly diseases. Each conversation should include the following information:
            1. Disease name (must include one of the following: Hypertension, Osteoarthritis or Rheumatoid Arthritis, Hyperlipidemia, Back Pain and Sciatica, Diabetes, Osteoporosis)
            2. Symptoms
            3. Diagnosis methods
            4. Treatment methods
            5. Prevention methods
            
            Each conversation should be in Korean and structured as a dialogue between a patient and a doctor. Please refer to the following conditions when creating the conversation:
            
            1. Write in Korean.
            2. The doctor asks various questions to get more detailed information from the patient.
            3. The patient answers the doctor's questions truthfully.
            4. Each exchange (question or answer) should be concise, preferably one sentence.
            5. The doctor provides a diagnosis based on the patient's answers.
            6. After the doctor's diagnosis, the patient asks how to cope or manage the condition.
            7. The doctor explains the treatment methods and prevention tips.
            8. The doctor initially knows nothing about the patient and gets all necessary information through questions and answers during the conversation.

            질병명: {row['disease_name']}
            
            환자: {row['symptoms']}
            의사: 증상이 언제부터 시작되었나요?
            환자: 한 달 전부터 시작됐어요.
            의사: 다른 증상은 없으신가요? 예를 들어, 잠을 잘 못 주무시거나, 피로감을 느끼시나요?
            환자: 가끔 {row['symptoms']} 외에 다른 증상이 있어요. 특히, 잠을 잘 못 자고 피로감을 느껴요.
            의사: 평소 혈압이나 혈당은 어떠신가요? 최근에 체중 변화가 있었나요?
            환자: 혈압은 조금 높고, 혈당은 정상 범위예요. 최근에 체중이 약간 줄었어요.
            의사: {row['diagnosis_methods']}
            환자: 네, 어떻게 치료해야 하나요?
            의사: {row['treatment_methods']}
            환자: 예방할 수 있는 방법이 있을까요?
            의사: {row['prevention_methods']} 그리고 규칙적인 건강 검진을 받는 것이 중요합니다.
            """
            prompts.append(prompt)
    return prompts

def generate_conversations_batch(batch_prompts):
    inputs = tokenizer(batch_prompts, return_tensors='pt', truncation=True, padding=True, max_length=1024).to(device)
    outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=150, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    conversations = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return conversations

# 질병 데이터
df_diseases = pd.DataFrame({
    "disease_name": ["고혈압", "골관절염", "류머티즘 관절염", "고지혈증", "요통 및 좌골신경통", "당뇨병", "골다공증", "치매"],
    "symptoms": ["최근에 두통이 자주 발생하고, 어지러움도 느껴요.", 
                 "무릎이 아프고 걸을 때 더 심해져요.", 
                 "아침에 손가락이 뻣뻣하고 통증이 있어요.", 
                 "최근 혈액 검사에서 콜레스테롤 수치가 높다고 나왔어요.", 
                 "허리가 아프고 다리까지 통증이 내려와요.", 
                 "최근에 갈증이 자주 나고 소변을 자주 봐요.", 
                 "최근에 뼈가 쉽게 부러지고 통증이 있어요.", 
                 "최근 기억력이 많이 떨어지고 길을 잃어버린 적이 있어요."],
    "diagnosis_methods": ["혈압을 측정하고 소변 검사와 혈액 검사를 통해 진단할 수 있습니다.", 
                          "관절 엑스레이와 MRI 검사를 통해 진단할 수 있습니다.", 
                          "혈액 검사와 X-ray 검사를 통해 진단할 수 있습니다.", 
                          "혈액 검사를 통해 고지혈증을 진단할 수 있습니다.", 
                          "MRI나 CT 스캔 검사를 통해 진단할 수 있습니다.", 
                          "혈당 검사를 통해 당뇨병을 진단할 수 있습니다.", 
                          "골밀도 검사를 통해 골다공증을 진단할 수 있습니다.", 
                          "인지 기능 검사와 뇌 스캔을 통해 진단할 수 있습니다."],
    "treatment_methods": ["약물 치료와 함께 식이 조절, 운동이 필요합니다.", 
                          "물리치료, 약물 치료, 심한 경우 수술을 고려할 수 있습니다.", 
                          "항염증제, 면역억제제 등의 약물 치료와 물리치료가 있습니다.", 
                          "식이요법, 운동, 약물 치료가 필요할 수 있어요.", 
                          "물리치료, 약물 치료, 심한 경우 수술을 고려할 수 있습니다.", 
                          "혈당 조절 약물 복용과 함께 식이 조절, 규칙적인 운동이 필요합니다.", 
                          "칼슘과 비타민 D 보충제, 규칙적인 운동, 약물 치료가 필요할 수 있습니다.", 
                          "약물 치료와 함께 인지 재활 치료가 필요합니다."],
    "prevention_methods": ["저염식 섭취, 정기적인 운동, 스트레스 관리를 통해 예방할 수 있습니다.", 
                           "적정 체중 유지, 규칙적인 운동, 관절에 무리가 가지 않게 하는 것이 중요합니다.", 
                           "정확한 예방 방법은 없지만, 조기 진단과 치료가 중요합니다.", 
                           "건강한 식습관, 규칙적인 운동, 금연이 도움이 됩니다.", 
                           "올바른 자세 유지, 규칙적인 운동, 무거운 물건 들기 피하기가 중요합니다.", 
                           "건강한 식습관 유지, 규칙적인 운동, 정기적인 혈당 체크가 중요합니다.", 
                           "칼슘과 비타민 D가 풍부한 음식 섭취, 규칙적인 운동, 금연과 절주가 도움이 됩니다.", 
                           "정기적인 인지 기능 검사와 두뇌 활동이 도움이 됩니다."]
})

# 데이터셋 생성 및 저장
total_conversations = []
batch_size = 10
prompts = create_prompts(df_diseases, 500)

for i in range(0, len(prompts), batch_size):
    batch_prompts = prompts[i:i+batch_size]
    conversations = generate_conversations_batch(batch_prompts)
    total_conversations.extend({'conversation': conv} for conv in conversations)
    print(f"Generated {len(total_conversations)} conversations so far")

df = pd.DataFrame(total_conversations)

# 데이터 라벨링
def label_data(conversation):
    lines = conversation.split('\n')
    labeled_data = []
    for line in lines:
        if '환자' in line:
          labeled_data.append({'role': 'patient', 'text': line})
        elif '의사' in line:
            labeled_data.append({'role': 'doctor', 'text': line})
    return labeled_data

# 라벨링된 데이터셋 생성
labeled_conversations = [label_data(conv['conversation']) for conv in total_conversations]
df_labeled = pd.DataFrame(labeled_conversations)

# 데이터셋으로 변환
dataset = Dataset.from_pandas(df_labeled)
dataset = dataset.train_test_split(test_size = .2)

# 데이터셋 저장
dataset.save_to_disk('C:/CareConnect/elderly_disease_conversations_labeled_dataset')

print('Data generation complete. Total entries:', len(total_conversations))