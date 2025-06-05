# Unsloth 4bit finetuning script for the Qwen3-0.6B voice command model

import os
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "unsloth/Qwen3-0.6B"

# Load model in 4bit and prepare LoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=2048,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# Combine JSONL files
DATA_DIR = "data"
jsonls = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".jsonl")]
combined_path = os.path.join(DATA_DIR, "combined_dataset.jsonl")
with open(combined_path, "w") as outfile:
    for p in jsonls:
        with open(p) as infile:
            for line in infile:
                outfile.write(line)

# Load dataset
raw_dataset = load_dataset("json", data_files={"train": combined_path})["train"]

def build_prompt(example):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": example["input"].strip()},
        {"role": "assistant", "content": example["output"].strip()},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

processed = raw_dataset.map(build_prompt)
train_test = processed.train_test_split(test_size=0.05, seed=42)

config = SFTConfig(
    dataset_text_field="text",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    logging_steps=10,
    save_steps=50,
    learning_rate=2e-4,
    report_to="none",
    optim="paged_adamw_8bit",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_test["train"],
    eval_dataset=train_test["test"],
    args=config,
)

trainer.train()

model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
