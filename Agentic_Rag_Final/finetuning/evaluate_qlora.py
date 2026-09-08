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

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. SETTINGS
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# EXISTING QLORA MODEL
# DO NOT CHANGE THIS PATH
QLORA_MODEL_PATH = "finetuning/output/qwen-qlora-v2"

# TEST DATASET
TEST_DATASET_PATH = "finetuning/data/test_dataset"

RESULTS_DIR = "finetuning/evaluation"

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "qlora_evaluation_results.json",
)

SCORE_FILE = os.path.join(
    RESULTS_DIR,
    "qlora_scores.json",
)

# Evaluate 20 questions from the held-out test dataset
NUM_TEST_QUESTIONS = 20

# Mentor passing threshold
PASS_THRESHOLD = 8.0

# Semantic similarity model
SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# 2. CHECK GPU
# ============================================================

print("=" * 60)
print("GPU INFORMATION")
print("=" * 60)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is not available."
    )

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

print()


# ============================================================
# 3. LOAD TEST DATASET
# ============================================================

print("=" * 60)
print("LOADING TEST DATASET")
print("=" * 60)

test_dataset = load_from_disk(
    TEST_DATASET_PATH
)

print(
    "Total test examples:",
    len(test_dataset)
)

print(
    "Questions used for this test:",
    min(
        NUM_TEST_QUESTIONS,
        len(test_dataset)
    )
)

print()


# ============================================================
# 4. LOAD TOKENIZER
# ============================================================

print("=" * 60)
print("LOADING TOKENIZER")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(
    "Tokenizer loaded successfully"
)

print()


# ============================================================
# 5. LOAD BASE MODEL IN 4-BIT
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

print(
    "Base Qwen loaded"
)

print()


# ============================================================
# 6. LOAD EXISTING QLORA ADAPTER
# ============================================================

print("=" * 60)
print("LOADING EXISTING QLORA ADAPTER")
print("=" * 60)

print(
    "Adapter path:",
    QLORA_MODEL_PATH
)

model = PeftModel.from_pretrained(
    base_model,
    QLORA_MODEL_PATH,
)

model.eval()

print()

print(
    "Fine-tuned QLoRA model loaded successfully!"
)

print()


# ============================================================
# 7. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# 8. LOAD SEMANTIC SIMILARITY MODEL
# ============================================================

print("=" * 60)
print("LOADING SEMANTIC SIMILARITY MODEL")
print("=" * 60)

semantic_model = SentenceTransformer(
    SEMANTIC_MODEL_NAME,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

print("Semantic model loaded successfully!")
print()

# ============================================================
# 8. GENERATE ANSWER
# ============================================================

def generate_answer(question):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful, accurate, and concise assistant. "
                "Understand the user's question carefully before answering. "
                "Give the correct answer directly. "
                "Do not guess or invent facts. "
                "If the question asks for a specific choice, "
                "give the correct choice first. "
                "If the question asks for a definition, "
                "give the standard meaning. "
                "Keep the answer focused on the question."
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

            # Shorter and more focused answers
            max_new_tokens=128,

            # Deterministic generation
            do_sample=False,

            # Reduce repeated text
            repetition_penalty=1.05,

            pad_token_id=tokenizer.pad_token_id,

            eos_token_id=tokenizer.eos_token_id,
        )

    # Remove input tokens
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
# 9. NORMALIZE TEXT
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

def calculate_exact_match(
    expected,
    generated
):

    expected_normalized = normalize_text(
        expected
    )

    generated_normalized = normalize_text(
        generated
    )

    return (
        expected_normalized
        ==
        generated_normalized
    )


# ============================================================
# 11. SEMANTIC SIMILARITY
# ============================================================

def calculate_semantic_similarity(
    expected,
    generated
):

    try:

        embeddings = semantic_model.encode(
            [expected, generated],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        similarity = torch.nn.functional.cosine_similarity(
            embeddings[0].unsqueeze(0),
            embeddings[1].unsqueeze(0),
        ).item()

        # Convert cosine similarity from [-1, 1] to [0, 10].
        score = ((similarity + 1) / 2) * 10

        return round(score, 2)

    except Exception as error:

        print("Semantic similarity error:", error)

        return 0.0


# ============================================================
# 12. START EVALUATION
# ============================================================

print("=" * 60)
print("STARTING QLORA EVALUATION")
print("=" * 60)

print()

print(
    "IMPORTANT:"
)

print(
    "The model will NOT be trained."
)

print(
    "The existing QLoRA adapter is only being evaluated."
)

print()


results = []

number_of_questions = min(
    NUM_TEST_QUESTIONS,
    len(test_dataset),
)


# ============================================================
# 13. EVALUATE QUESTIONS
# ============================================================

for index in range(
    number_of_questions
):

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

    print(
        "GENERATING QLORA ANSWER..."
    )

    generated_answer = generate_answer(
        question
    )

    print()

    print("QLORA ANSWER:")

    print(generated_answer)

    print()

    print(
        "EXPECTED DATASET ANSWER:"
    )

    print(expected_answer)

    # --------------------------------------------------------
    # Exact match
    # --------------------------------------------------------

    exact_match = calculate_exact_match(
        expected_answer,
        generated_answer
    )

    # --------------------------------------------------------
    # Semantic similarity
    # --------------------------------------------------------

    similarity_score = calculate_semantic_similarity(
        expected_answer,
        generated_answer
    )

    print()

    print(
        "EXACT MATCH:",
        exact_match
    )

    print(
        "SEMANTIC SIMILARITY:",
        similarity_score,
        "/ 10"
    )

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    result = {

        "question_id":
            index + 1,

        "question":
            question,

        "expected_answer":
            expected_answer,

        "qlora_answer":
            generated_answer,

        "exact_match":
            exact_match,

        "similarity_score":
            similarity_score,
    }

    results.append(
        result
    )

    print()

    print(
        "Question evaluation completed."
    )

    print()


# ============================================================
# 14. SAVE QUESTION RESULTS
# ============================================================

print("=" * 60)

print(
    "SAVING EVALUATION RESULTS"
)

print("=" * 60)

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
# 15. CALCULATE SUMMARY
# ============================================================

exact_matches = sum(
    1
    for result in results
    if result["exact_match"]
)

if not results:
    raise RuntimeError("No evaluation results were generated.")

average_similarity = (
    sum(
        result["similarity_score"]
        for result in results
    )
    / len(results)
)


average_similarity = round(
    average_similarity,
    2
)


# ============================================================
# 16. PASS / FAIL
# ============================================================

# IMPORTANT:
# This is only a text-similarity-based indicator.
# It is NOT a factual correctness score.

if average_similarity > PASS_THRESHOLD:

    result_status = "PASS"

else:

    result_status = "FAIL"


# ============================================================
# 17. SAVE SCORE
# ============================================================

score_data = {

    "model":
        QLORA_MODEL_PATH,

    "test_dataset":
        TEST_DATASET_PATH,

    "questions_evaluated":
        len(results),

    "exact_matches":
        exact_matches,

    "average_semantic_similarity":
        average_similarity,

    "pass_threshold":
        PASS_THRESHOLD,

    "status":
        result_status,
}


with open(
    SCORE_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        score_data,
        file,
        indent=4,
    )


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print()

print("=" * 60)

print(
    "QLORA EVALUATION SUMMARY"
)

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
    "Average semantic similarity:",
    average_similarity,
    "/ 10"
)

print(
    "Pass threshold (must be >):",
    PASS_THRESHOLD,
    "/ 10"
)

print()

print(
    "RESULT:",
    result_status
)

print()

print(
    "Question results:"
)

print(
    RESULTS_FILE
)

print()

print(
    "Score file:"
)

print(
    SCORE_FILE
)

print()

print("=" * 60)

print(
    "IMPORTANT"
)

print("=" * 60)

print(
    "No training was performed."
)

print(
    "The existing qwen-qlora model was only loaded and tested."
)

print(
    "The 8/10 value is a text-similarity threshold, "
    "not a factual correctness measurement."
)

print()