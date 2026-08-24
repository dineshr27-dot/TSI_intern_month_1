import os
import json
import pandas as pd

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please add it to your .env file.")

print("GROQ API key loaded")


# Load prompts from files
def load_prompt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


prompt_v1_text = load_prompt("prompts/prompt_v1.txt")
prompt_v2_text = load_prompt("prompts/prompt_v2.txt")

print("Prompt V1 loaded")
print("Prompt V2 loaded")


# Create LangChain prompt templates
prompt_v1 = ChatPromptTemplate.from_template(prompt_v1_text)
prompt_v2 = ChatPromptTemplate.from_template(prompt_v2_text)

print("LangChain prompt templates created")


# Create Groq model
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

print("Groq model initialized")
print("Model: openai/gpt-oss-20b")


# Load test questions
with open("data/test_queries.json", "r", encoding="utf-8") as file:
    test_queries = json.load(file)

print(f"Loaded {len(test_queries)} test queries")


# Store results
results = []


# Run A/B test
for item in test_queries:

    query_id = item["id"]
    question = item["question"]
    context = item["context"]

    print("\n" + "=" * 60)
    print(f"Query {query_id}")
    print("=" * 60)
    print(f"Question: {question}")

    # Run Prompt V1
    print("\nRunning Prompt V1...")

    messages_v1 = prompt_v1.invoke({
        "context": context,
        "question": question
    })

    response_v1 = llm.invoke(messages_v1)
    answer_v1 = response_v1.content

    # Run Prompt V2
    print("Running Prompt V2...")

    messages_v2 = prompt_v2.invoke({
        "context": context,
        "question": question
    })

    response_v2 = llm.invoke(messages_v2)
    answer_v2 = response_v2.content

    # Display answers
    print("\nV1 Answer:")
    print(answer_v1)

    print("\nV2 Answer:")
    print(answer_v2)

    # Store results
    results.append({
        "query_id": query_id,
        "question": question,
        "context": context,
        "v1_answer": answer_v1,
        "v2_answer": answer_v2
    })


# Create DataFrame
df = pd.DataFrame(results)


# Create results folder
os.makedirs("results", exist_ok=True)


# Save results
output_file = "results/ab_test_results.csv"

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8"
)


# Final message
print("\n" + "=" * 60)
print("A/B TEST COMPLETED")
print("=" * 60)

print(f"\nTotal queries tested: {len(test_queries)}")

print("\nModel:")
print("openai/gpt-oss-20b")

print("\nPrompts:")
print("V1 -> prompts/prompt_v1.txt")
print("V2 -> prompts/prompt_v2.txt")

print("\nResults saved to:")
print(output_file)

print("\nNext step:")
print("Evaluate V1 and V2 answers using evaluate.py")