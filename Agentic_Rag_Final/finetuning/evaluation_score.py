import json
import os

import matplotlib.pyplot as plt
import torch
from sentence_transformers import SentenceTransformer


# ============================================================
# SETTINGS
# ============================================================

RESULTS_FILE = (
    "finetuning/evaluation/"
    "qlora_evaluation_results.json"
)

OUTPUT_DIR = "finetuning/evaluation"

CHART_FILE = os.path.join(
    OUTPUT_DIR,
    "qlora_evaluation_chart.png"
)

SCORE_FILE = os.path.join(
    OUTPUT_DIR,
    "qlora_scores.json"
)

PASS_THRESHOLD = 8.0

# Semantic similarity model
SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# LOAD RESULTS
# ============================================================

print("=" * 60)
print("LOADING QLORA EVALUATION RESULTS")
print("=" * 60)

with open(
    RESULTS_FILE,
    "r",
    encoding="utf-8"
) as file:
    results = json.load(file)

print("Results loaded:", len(results))
print()


# ============================================================
# LOAD SEMANTIC MODEL
# ============================================================

print("=" * 60)
print("LOADING SEMANTIC SIMILARITY MODEL")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)
print("Model:", SEMANTIC_MODEL)

semantic_model = SentenceTransformer(
    SEMANTIC_MODEL,
    device=device
)

print("Semantic model loaded successfully.")
print()


# ============================================================
# CALCULATE SEMANTIC SIMILARITY
# ============================================================

def calculate_similarity(expected, generated):

    if not expected or not generated:
        return 0.0

    expected_embedding = semantic_model.encode(
        expected,
        convert_to_tensor=True
    )

    generated_embedding = semantic_model.encode(
        generated,
        convert_to_tensor=True
    )

    similarity = torch.nn.functional.cosine_similarity(
        expected_embedding.unsqueeze(0),
        generated_embedding.unsqueeze(0)
    ).item()

    # Convert cosine similarity (-1 to +1)
    # into score (0 to 10)
    score = ((similarity + 1) / 2) * 10

    # Keep score between 0 and 10
    score = max(0.0, min(10.0, score))

    return score


# ============================================================
# EVALUATE EACH QUESTION
# ============================================================

print("=" * 60)
print("CALCULATING SEMANTIC SIMILARITY SCORES")
print("=" * 60)

scores = []

exact_matches = 0

for result in results:

    question_id = result["question_id"]

    question = result["question"]

    expected = result["expected_answer"]

    generated = result["qlora_answer"]

    score = calculate_similarity(
        expected,
        generated
    )

    scores.append(score)

    # Exact match check
    if (
        expected.strip().lower()
        == generated.strip().lower()
    ):
        exact_matches += 1

    print(
        f"Question {question_id}: "
        f"{score:.2f}/10"
    )


# ============================================================
# OVERALL SCORE
# ============================================================

average_score = (
    sum(scores) / len(scores)
)

if average_score > PASS_THRESHOLD:
    final_result = "PASS"
else:
    final_result = "FAIL"


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 60)
print("FINAL QLORA EVALUATION")
print("=" * 60)

print(
    f"Questions evaluated : {len(scores)}"
)

print(
    f"Exact matches       : "
    f"{exact_matches}/{len(scores)}"
)

print(
    f"Average semantic similarity : "
    f"{average_score:.2f}/10"
)

print(
    f"Pass threshold      : "
    f">{PASS_THRESHOLD:.1f}/10"
)

print(
    f"Result              : "
    f"{final_result}"
)

print()


# ============================================================
# SAVE SCORES
# ============================================================

score_results = {

    "model": "Fine-tuned Qwen QLoRA",

    "evaluation_method": (
        "SentenceTransformer semantic similarity"
    ),

    "semantic_model": SEMANTIC_MODEL,

    "questions_evaluated": len(scores),

    "exact_matches": exact_matches,

    "average_semantic_similarity": average_score,

    "pass_threshold": PASS_THRESHOLD,

    "result": final_result,

    "individual_scores": scores,
}


with open(
    SCORE_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        score_results,
        file,
        indent=4
    )


# ============================================================
# MATPLOTLIB CHART
# ============================================================

print("=" * 60)
print("CREATING MATPLOTLIB CHART")
print("=" * 60)

question_numbers = list(
    range(1, len(scores) + 1)
)

plt.figure(
    figsize=(12, 6)
)

plt.bar(
    question_numbers,
    scores
)

plt.axhline(
    y=PASS_THRESHOLD,
    linestyle="--",
    label="Pass threshold (8/10)"
)

plt.xlabel(
    "Question Number"
)

plt.ylabel(
    "Semantic Similarity Score / 10"
)

plt.title(
    "Fine-tuned Qwen QLoRA Evaluation"
)

plt.xticks(
    question_numbers
)

plt.ylim(
    0,
    10
)

plt.legend()

plt.tight_layout()

plt.savefig(
    CHART_FILE,
    dpi=200
)

plt.close()


# ============================================================
# COMPLETED
# ============================================================

print()
print("=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)

print()

print("Score file:")
print(SCORE_FILE)

print()

print("Matplotlib chart:")
print(CHART_FILE)

print()

print(
    "Evaluation only."
)

print(
    "No training was performed."
)

print(
    "The existing QLoRA model was not modified."
)

print()

print(
    "IMPORTANT:"
)

print(
    "The semantic similarity score measures "
    "meaning similarity between expected and generated answers."
)

print(
    "It is not a direct measure of factual correctness."
)