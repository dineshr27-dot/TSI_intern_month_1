import json
import os

# ============================================================
# CORRECTNESS EVALUATION
# ============================================================

RESULTS_FILE = os.path.join(
    "finetuning",
    "evaluation",
    "qlora_evaluation_results.json"
)

OUTPUT_FILE = os.path.join(
    "finetuning",
    "evaluation",
    "correctness_results.json"
)


def get_value(item, possible_keys):
    """Find a value using multiple possible JSON key names."""
    for key in possible_keys:
        if key in item:
            return item[key]
    return ""


# ------------------------------------------------------------
# LOAD EXISTING EVALUATION RESULTS
# ------------------------------------------------------------

if not os.path.exists(RESULTS_FILE):
    print("❌ Evaluation results file not found:")
    print(RESULTS_FILE)
    print()
    print("First run:")
    print("python finetuning\\evaluate_qlora.py")
    exit()


with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


# ------------------------------------------------------------
# HANDLE DIFFERENT JSON FORMATS
# ------------------------------------------------------------

if isinstance(data, dict):

    if "results" in data:
        results = data["results"]

    elif "evaluations" in data:
        results = data["evaluations"]

    elif "data" in data:
        results = data["data"]

    else:
        # If dictionary itself contains question records
        results = list(data.values())

elif isinstance(data, list):
    results = data

else:
    print("❌ Unknown JSON format.")
    exit()


print("=" * 70)
print("QLoRA CORRECTNESS EVALUATION")
print("=" * 70)

print(f"\nQuestions found: {len(results)}")

if len(results) == 0:
    print("❌ No evaluation results found.")
    exit()


# ------------------------------------------------------------
# EVALUATE EACH ANSWER
# ------------------------------------------------------------

evaluated_results = []

total_score = 0

for index, item in enumerate(results, start=1):

    question = get_value(
        item,
        [
            "question",
            "prompt",
            "instruction",
            "input"
        ]
    )

    expected = get_value(
        item,
        [
            "expected_answer",
            "expected",
            "reference_answer",
            "reference",
            "ground_truth",
            "answer"
        ]
    )

    generated = get_value(
        item,
        [
            "generated_answer",
            "generated",
            "model_answer",
            "response",
            "output"
        ]
    )

    print("\n" + "=" * 70)
    print(f"QUESTION {index}")
    print("=" * 70)

    print("\nQuestion:")
    print(question)

    print("\nExpected Answer:")
    print(expected)

    print("\nQLoRA Generated Answer:")
    print(generated)

    print("\n----------------------------------------")
    print("Give correctness score from 0 to 10")
    print("----------------------------------------")
    print("10 = Completely correct")
    print("8  = Correct with minor issue")
    print("6  = Mostly correct")
    print("4  = Partially correct")
    print("2  = Mostly incorrect")
    print("0  = Completely wrong")

    while True:
        try:
            score = float(input(f"\nScore for Q{index} (0-10): "))

            if 0 <= score <= 10:
                break

            print("❌ Enter a number between 0 and 10.")

        except ValueError:
            print("❌ Please enter a valid number.")

    reason = input("Reason: ").strip()

    total_score += score

    evaluated_results.append(
        {
            "question_id": index,
            "question": question,
            "expected_answer": expected,
            "generated_answer": generated,
            "correctness_score": score,
            "reason": reason
        }
    )


# ------------------------------------------------------------
# CALCULATE AVERAGE
# ------------------------------------------------------------

average_score = total_score / len(evaluated_results)


if average_score >= 8.0:
    result = "PASS"
else:
    result = "FAIL"


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

final_report = {
    "evaluation_type": "Manual Correctness Evaluation",
    "questions_evaluated": len(evaluated_results),
    "average_correctness": round(average_score, 2),
    "pass_threshold": 8.0,
    "result": result,
    "results": evaluated_results
}


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        final_report,
        f,
        indent=4,
        ensure_ascii=False
    )


# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

print("\n\n")
print("=" * 70)
print("FINAL CORRECTNESS EVALUATION")
print("=" * 70)

print(f"Questions evaluated : {len(evaluated_results)}")
print(f"Average correctness : {average_score:.2f} / 10")
print(f"Pass threshold      : 8.00 / 10")
print(f"RESULT              : {result}")

print("=" * 70)

print("\nSaved report:")
print(OUTPUT_FILE)

print("\n✅ No training was performed.")
print("✅ Existing QLoRA model was not changed.")
print("✅ This evaluation measures actual answer correctness.")