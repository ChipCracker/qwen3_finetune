# Qwen3-0.6B Voice Command Finetuning

This repository provides code and instructions for supervised finetuning of a Qwen3-0.6B language model for voice command interpretation tasks. The goal is to adapt the LLM for domain-specific understanding and response to spoken instructions, using a conversational prompt format compatible with Qwen-Chat models.

---

## Use in ROS2

This model is intended for use with the [robo_voice_control](https://github.com/ChipCracker/robo-voice-control) project. After finetuning, integrate the checkpoint in your ROS2 node for LLM-based command interpretation. For example, set the model_path parameter of your llm_node to the exported model directory or checkpoint, and ensure the prompt format matches the Qwen chat template as described above.

## Features

* **Qwen3-0.6B** base model support (swap out for any Qwen2/Qwen3 variant as needed)
* **Instruction/Chat-style finetuning** (user input, assistant output)
* **Custom voice command dataset (JSONL format)**
* Evaluation/train split
* HuggingFace Trainer and Datasets workflow

---

## Data Format

Training data must be in JSONL format, e.g.:

```jsonl
{"input": "Licht an", "output": "Turning on the light."}
{"input": "Starte Musik", "output": "Playing music."}
```

## Prompt Format (Qwen Chat)

Each sample is converted to Qwen chat prompt style:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
```


---

## Finetuning Tips & Recommendations

* Always match prompt formatting to your Qwen version.
* For longer commands or richer conversations, adjust `max_length` accordingly.
* Use `fp16` or `bf16` if your hardware supports it.
* For larger data or longer training, adjust `save_steps`, `logging_steps`, and \`eval\_steps

## Training with Unsloth 4bit

Run the provided `finetune_unsloth_qwen3.py` script to finetune Qwen3 using Unsloth's 4bit quantization. The script loads all JSONL files in the `data` directory, formats them with the Qwen chat template and performs LoRA finetuning in 4bit precision.

```bash
python finetune_unsloth_qwen3.py
```

The resulting adapters are stored in the `lora_model` directory.

