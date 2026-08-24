import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# -----------------------------
# 1. Load tokenizer
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# -----------------------------
# 2. 4-bit QLoRA configuration
# -----------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# -----------------------------
# 3. Load base model
# -----------------------------

print("Loading TinyLlama in 4-bit...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

print("Base model loaded!")


# -----------------------------
# 4. LoRA configuration
# -----------------------------

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)


# -----------------------------
# 5. Attach LoRA
# -----------------------------

print("\nAttaching LoRA adapter...")

model = get_peft_model(model, lora_config)

print("LoRA adapter attached successfully!")


# -----------------------------
# 6. Show trainable parameters
# -----------------------------

print("\nTrainable parameters:")
model.print_trainable_parameters()