import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "./tinyllama-lora"


# ============================================================
# GPU
# ============================================================

print("=" * 60)
print("LOADING MODEL")
print("=" * 60)

print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# 4-BIT CONFIG
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("\nLoading base TinyLlama...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
)

print("Base model loaded!")


# ============================================================
# LOAD LoRA MODEL
# ============================================================

print("\nLoading LoRA adapter...")

finetuned_model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)

finetuned_model.eval()

print("Fine-tuned model loaded!")


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
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Remove the question/prompt tokens
    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return answer.strip()


# ============================================================
# USER QUESTION LOOP
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print("\nType your question.")
print("Type 'exit' to stop.")

while True:

    print("\n" + "-" * 60)

    question = input("User: ")

    if question.lower().strip() == "exit":
        print("\nExiting...")
        break

    if not question.strip():
        print("Please enter a question.")
        continue


    # ========================================================
    # BASE MODEL
    # ========================================================

    print("\n[BASE MODEL]")
    print("-" * 40)

    base_answer = generate_answer(
        base_model,
        question,
    )

    print(base_answer)


    # ========================================================
    # FINE-TUNED MODEL
    # ========================================================

    print("\n[FINE-TUNED MODEL]")
    print("-" * 40)

    finetuned_answer = generate_answer(
        finetuned_model,
        question,
    )

    print(finetuned_answer)