import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset

# Choose your Qwen2/Qwen3 model
MODEL_NAME = "Qwen/Qwen3-0.6B"  # Replace with the exact model name, e.g., "Qwen/Qwen3-7B-Chat" if available

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Load your data
dataset = load_dataset("json", data_files={"train": "data/voice_command_dataset.jsonl"})

def make_qwen_chat_prompt(example):
    # Bilde Prompt im Qwen2-Chat Stil (siehe oben)
    input_text = example["input"].strip()
    output_text = example["output"].strip()
    return {
        "text": (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{input_text}<|im_end|>\n"
            f"<|im_start|>assistant\n{output_text}<|im_end|>\n"
        )
    }

dataset = dataset["train"].map(make_qwen_chat_prompt)

# Split für eval
split = dataset.train_test_split(test_size=0.05)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)

tokenized_datasets = split.map(tokenize_function, batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="/Volumes/DATA SSD/LLMs/finetunes/qwen3_0.6B",
    #evaluation_strategy="steps",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    save_steps=100,
    logging_steps=10,
    eval_steps=100,
    fp16=False,  # oder bf16
    save_total_limit=2,
    learning_rate=2e-5,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
)

trainer.train()