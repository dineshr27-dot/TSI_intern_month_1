import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel


# ============================================================
# 1. SETTINGS
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ADAPTER_PATH = "finetuning/output/qwen-qlora"


# ============================================================
# 2. CHECK GPU
# ============================================================

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 3. 4-BIT CONFIGURATION
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# 4. LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)


# ============================================================
# 5. LOAD BASE MODEL
# ============================================================

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)


# ============================================================
# 6. LOAD TRAINED LoRA ADAPTER
# ============================================================

print("Loading fine-tuned LoRA adapter...")

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH,
)

model.eval()

print("Fine-tuned model loaded successfully!")


# ============================================================
# 7. ASK QUESTION
# ============================================================

question = input("\nEnter your question: ")


prompt = f"""### Instruction:
{question}

### Response:
"""


# ============================================================
# 8. TOKENIZE
# ============================================================

inputs = tokenizer(
    prompt,
    return_tensors="pt",
).to(model.device)


# ============================================================
# 9. GENERATE RESPONSE
# ============================================================

print("\nGenerating response...\n")

with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
    )


# ============================================================
# 10. DECODE
# ============================================================

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True,
)


# ============================================================
# 11. PRINT
# ============================================================

print("=" * 60)
print("FINE-TUNED MODEL RESPONSE")
print("=" * 60)

print(response)