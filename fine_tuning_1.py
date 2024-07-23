import os
import torch
from transformers import LlamaForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model

# 환경 변수
os.environ['HUGGINGFACE_API_KEY'] = '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'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# MODEL_AND_TOKENiZER
model_name = 'squarelike/llama2-ko-medical-7b'
tokenizer = AutoTokenizer.from_pretrained(model_name, token = os.environ['HUGGINGFACE_API_KEY'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LlamaForCausalLM.from_pretrained(model_name, token = os.environ['HUGGINGFACE_API_KEY']).to(device)
tokenizer.pad_token = tokenizer.eos_token

# QLoRA
lora_config = LoraConfig(
    lora_alpha = 16,
    lora_dropout = .01,
    r = 64,
    target_modules = ['q_proj', 'v_proj']
)

model = get_peft_model(model, lora_config)

# Dataset
dataset_path = 'C:/CareConnect/elderly_disease_conversations_labeled_dataset'
dataset = load_from_disk(dataset_path)
train_dataset = dataset['train']
test_dataset = dataset['test']

# 데이터셋의 일부 출력
print("Train dataset sample:")
print(train_dataset[0])

print("\nTest dataset sample:")
print(test_dataset[0])

# TrainingArguments for fine-tuning
training_args = TrainingArguments(
    output_dir = './results',
    evaluation_strategy = 'epoch',
    save_strategy = 'epoch',
    learning_rate = 3e-4,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    num_train_epochs = 3,
    weight_decay = .01,
    push_to_hub = False,
    logging_dir = './logs',
    logging_steps = 10,
    save_total_limit = 2,
    load_best_model_at_end = True
)

# Trainer
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = test_dataset
)

# Training Model
trainer.train()

# Save Model
trainer.save_model('./fine_tuned_llama2')
tokenizer.save_pretrained('./fine_tuned_llama2')