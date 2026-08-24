import pandas as pd
import re


# Load evaluation results
df = pd.read_csv("results/evaluation_results.csv")


v1_scores = []
v2_scores = []


# Extract scores from each evaluation
for evaluation in df["evaluation"]:

    v1_match = re.search(
        r"V1:.*?Overall:\s*(\d+)",
        evaluation,
        re.DOTALL
    )

    v2_match = re.search(
        r"V2:.*?Overall:\s*(\d+)",
        evaluation,
        re.DOTALL
    )

    if v1_match and v2_match:
        v1_scores.append(int(v1_match.group(1)))
        v2_scores.append(int(v2_match.group(1)))


# Calculate averages
v1_average = sum(v1_scores) / len(v1_scores)
v2_average = sum(v2_scores) / len(v2_scores)


print("=" * 50)
print("A/B TEST SUMMARY")
print("=" * 50)

print(f"\nNumber of queries: {len(v1_scores)}")

print(f"\nPrompt V1 average score: {v1_average:.2f}/5")
print(f"Prompt V2 average score: {v2_average:.2f}/5")


# Find winner
if v1_average > v2_average:
    print("\nWinner: Prompt V1")

elif v2_average > v1_average:
    print("\nWinner: Prompt V2")

else:
    print("\nResult: Tie")


print("\nScore comparison:")
print(f"V1: {v1_scores}")
print(f"V2: {v2_scores}")