import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel


# ============================================================
# SETTINGS
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ADAPTER_PATH = "finetuning/output/qwen-qlora"


# ============================================================
# GPU
# ============================================================

print("=" * 60)
print("GPU INFORMATION")
print("=" * 60)

print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available.")

print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded!")


# ============================================================
# 4-BIT CONFIGURATION
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(model, question):

    prompt = f"""### Instruction:
{question}

### Response:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_tokens = outputs[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return answer.strip()


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("\nLoading BASE Qwen model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

base_model.eval()

print("Base model loaded!")


# ============================================================
# LOAD FINE-TUNED MODEL
# ============================================================

print("\nLoading FINE-TUNED QLoRA model...")

fine_tuned_model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)

fine_tuned_model.eval()

print("Fine-tuned model loaded!")


# ============================================================
# INTERACTIVE COMPARISON
# ============================================================

print("\n")
print("=" * 60)
print("BASE MODEL vs FINE-TUNED MODEL")
print("=" * 60)

print("\nType your question.")
print("Type 'exit' to stop.")


while True:

    question = input("\nEnter your question: ")

    if question.lower().strip() == "exit":
        break

    if not question.strip():
        print("Please enter a question.")
        continue


    # ========================================================
    # BASE MODEL
    # ========================================================

    print("\n" + "-" * 60)
    print("BASE QWEN MODEL")
    print("-" * 60)

    base_answer = generate_answer(
        base_model,
        question,
    )

    print(base_answer)


    # ========================================================
    # FINE-TUNED MODEL
    # ========================================================

    print("\n" + "-" * 60)
    print("FINE-TUNED QWEN + LoRA")
    print("-" * 60)

    fine_tuned_answer = generate_answer(
        fine_tuned_model,
        question,
    )

    print(fine_tuned_answer)


    # ========================================================
    # COMPARISON END
    # ========================================================

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)