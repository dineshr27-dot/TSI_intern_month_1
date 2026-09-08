import os
import json
import re
import torch

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. SETTINGS
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# EXISTING MODEL — DO NOT CHANGE
QLORA_MODEL_PATH = "finetuning/output/qwen-qlora"

# TEST DATASET — NEVER USED FOR TRAINING
TEST_DATASET_PATH = "finetuning/data/test_dataset"

RESULTS_DIR = "finetuning/evaluation"

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "qlora_evaluation_results.json",
)

NUM_TEST_QUESTIONS = 10


# ============================================================
# 2. CHECK GPU
# ============================================================

print("=" * 60)
print("GPU INFORMATION")
print("=" * 60)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available.")

print("GPU:", torch.cuda.get_device_name(0))
print()


# ============================================================
# 3. LOAD TEST DATASET
# ============================================================

print("=" * 60)
print("LOADING TEST DATASET")
print("=" * 60)

test_dataset = load_from_disk(TEST_DATASET_PATH)

print("Total test examples:", len(test_dataset))
print(
    "Questions used:",
    min(NUM_TEST_QUESTIONS, len(test_dataset))
)
print()


# ============================================================
# 4. LOAD TOKENIZER
# ============================================================

print("=" * 60)
print("LOADING TOKENIZER")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded successfully")
print()


# ============================================================
# 5. LOAD BASE MODEL
# ============================================================

print("=" * 60)
print("LOADING BASE QWEN")
print("=" * 60)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

print("Base Qwen loaded")
print()


# ============================================================
# 6. LOAD EXISTING QLORA
# ============================================================

print("=" * 60)
print("LOADING EXISTING QLORA")
print("=" * 60)

print("Adapter:", QLORA_MODEL_PATH)

model = PeftModel.from_pretrained(
    base_model,
    QLORA_MODEL_PATH,
)

model.eval()

print()
print("Fine-tuned QLoRA loaded successfully!")
print()


# ============================================================
# 7. OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# 8. GENERATE ANSWER
# ============================================================

def generate_answer(question):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful and accurate assistant. "
                "Answer the user's question directly and clearly. "
                "Do not add unnecessary information."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

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
            max_new_tokens=128,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return answer.strip()


# ============================================================
# 9. TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    text = text.lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    return text


# ============================================================
# 10. EXACT MATCH
# ============================================================

def exact_match(expected, generated):

    return (
        normalize_text(expected)
        ==
        normalize_text(generated)
    )


# ============================================================
# 11. TF-IDF SEMANTIC-LIKE SIMILARITY
# ============================================================

def calculate_similarity(expected, generated):

    try:

        vectorizer = TfidfVectorizer()

        vectors = vectorizer.fit_transform(
            [expected, generated]
        )

        score = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0]

        return round(score * 10, 2)

    except Exception:

        return 0.0


# ============================================================
# 12. EVALUATE
# ============================================================

print("=" * 60)
print("STARTING QLORA EVALUATION")
print("=" * 60)

print()
print("NO TRAINING WILL BE PERFORMED.")
print("EXISTING QLORA MODEL ONLY.")
print()


results = []

number_of_questions = min(
    NUM_TEST_QUESTIONS,
    len(test_dataset),
)


for index in range(number_of_questions):

    example = test_dataset[index]

    question = example["instruction"]
    expected_answer = example["response"]

    print("=" * 60)
    print(
        f"QUESTION {index + 1}/{number_of_questions}"
    )
    print("=" * 60)

    print()
    print("QUESTION:")
    print(question)

    print()
    print("GENERATING ANSWER...")

    generated_answer = generate_answer(
        question
    )

    print()
    print("QLORA ANSWER:")
    print(generated_answer)

    print()
    print("EXPECTED ANSWER:")
    print(expected_answer)

    # Exact match
    is_exact = exact_match(
        expected_answer,
        generated_answer
    )

    # Similarity
    similarity_score = calculate_similarity(
        expected_answer,
        generated_answer
    )

    print()
    print("EXACT MATCH:", is_exact)
    print(
        "TEXT SIMILARITY:",
        similarity_score,
        "/ 10"
    )

    result = {
        "question_id": index + 1,
        "question": question,
        "expected_answer": expected_answer,
        "qlora_answer": generated_answer,
        "exact_match": is_exact,
        "similarity_score": similarity_score,
    }

    results.append(result)

    print()


# ============================================================
# 13. SAVE RESULTS
# ============================================================

with open(
    RESULTS_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        results,
        file,
        indent=4,
        ensure_ascii=False,
    )


# ============================================================
# 14. SUMMARY
# ============================================================

exact_matches = sum(
    1
    for result in results
    if result["exact_match"]
)

average_similarity = (
    sum(
        result["similarity_score"]
        for result in results
    )
    / len(results)
)


print("=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)

print()

print(
    "Questions evaluated:",
    len(results)
)

print(
    "Exact matches:",
    exact_matches,
    "/",
    len(results)
)

print(
    "Average text similarity:",
    round(average_similarity, 2),
    "/ 10"
)

print()

print("Results saved to:")
print(RESULTS_FILE)

print()

print("IMPORTANT:")
print("No training was performed.")
print("Existing qwen-qlora was only evaluated.")