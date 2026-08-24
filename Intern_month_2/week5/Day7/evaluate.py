import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os


# Load environment variables
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in .env file")


# Load the A/B test results
df = pd.read_csv("results/ab_test_results.csv")

print(f"Loaded {len(df)} test results")


# Create the evaluator model
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# Evaluation prompt
evaluation_prompt = ChatPromptTemplate.from_template("""
You are evaluating two answers given by an AI.

Question:
{question}

Context:
{context}

Answer V1:
{v1_answer}

Answer V2:
{v2_answer}

Evaluate both answers.

Give each answer a score from 1 to 5 for:

1. Accuracy
2. Relevance
3. Groundedness
4. Overall Quality

Use this format exactly:

V1:
Accuracy: X
Relevance: X
Groundedness: X
Overall: X

V2:
Accuracy: X
Relevance: X
Groundedness: X
Overall: X
""")


results = []


# Evaluate every query
for _, row in df.iterrows():

    print("\n" + "=" * 60)
    print(f"Evaluating Query {row['query_id']}")
    print("=" * 60)

    prompt = evaluation_prompt.invoke({
        "question": row["question"],
        "context": row["context"],
        "v1_answer": row["v1_answer"],
        "v2_answer": row["v2_answer"]
    })

    response = llm.invoke(prompt)

    evaluation = response.content

    print(evaluation)

    results.append({
        "query_id": row["query_id"],
        "evaluation": evaluation
    })


# Save evaluation results
evaluation_df = pd.DataFrame(results)

evaluation_df.to_csv(
    "results/evaluation_results.csv",
    index=False
)


print("\n" + "=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)

print("\nResults saved to:")
print("results/evaluation_results.csv")