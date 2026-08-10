# ============================================================
# WEEK 4 - DAY 1
# TRANSFORMERS, TOKENS, PROMPTING & CHAT ROLES
# ============================================================

# ============================================================
# 1. TOKENS - BASIC INTUITION
# ============================================================

text = "Machine learning is useful."

# Simple demonstration only.
# Real LLM tokenizers are more sophisticated.
tokens = text.split()

print("Original text:")
print(text)

print("\nSimple tokens:")
print(tokens)

print("\nNumber of simple tokens:", len(tokens))


# ============================================================
# 2. TOKENIZATION USING HUGGING FACE
# ============================================================

# Uncomment if transformers is not installed:
# !pip install transformers

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

text = "Machine learning is useful."

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print("\n===================================")
print("HUGGING FACE TOKENIZATION")
print("===================================")

print("Text:")
print(text)

print("\nTokens:")
print(tokens)

print("\nToken IDs:")
print(token_ids)


# ============================================================
# 3. ATTENTION - INTUITION
# ============================================================

print("\n===================================")
print("ATTENTION INTUITION")
print("===================================")

sentence = (
    "The animal didn't cross the road "
    "because it was tired."
)

print(sentence)

print("""
Attention allows a transformer to consider
relationships between different tokens.

For example:

"The animal ... because it was tired."

When processing "it", the model can use
information from surrounding tokens to determine
what "it" refers to.

This is the basic intuition behind attention.
""")


# ============================================================
# 4. ZERO-SHOT PROMPTING
# ============================================================

zero_shot_prompt = """
Classify the following sentence as Positive or Negative:

"I really enjoyed this movie."
"""

print("\n===================================")
print("ZERO-SHOT PROMPTING")
print("===================================")

print(zero_shot_prompt)


# ============================================================
# 5. FEW-SHOT PROMPTING
# ============================================================

few_shot_prompt = """
Classify the sentiment.

Example 1:
Text: "I love this phone."
Sentiment: Positive

Example 2:
Text: "This phone is terrible."
Sentiment: Negative

Now classify:

Text: "The phone works perfectly."
Sentiment:
"""

print("\n===================================")
print("FEW-SHOT PROMPTING")
print("===================================")

print(few_shot_prompt)


# ============================================================
# 6. CHAIN-OF-THOUGHT / REASONING PROMPT
# ============================================================

reasoning_prompt = """
Solve the following problem and provide a concise
explanation of the key steps.

If a product costs 1000 and has a 20% discount,
what is the final price?
"""

print("\n===================================")
print("REASONING PROMPT")
print("===================================")

print(reasoning_prompt)


# ============================================================
# 7. SYSTEM / USER / ASSISTANT ROLES
# ============================================================

messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful SQL tutor. "
            "Explain concepts using simple examples."
        )
    },
    {
        "role": "user",
        "content": "What is an INNER JOIN?"
    },
    {
        "role": "assistant",
        "content": (
            "An INNER JOIN returns rows where "
            "there is a matching value in both tables."
        )
    }
]

print("\n===================================")
print("CHAT ROLES")
print("===================================")

for message in messages:
    print("\nRole:", message["role"])
    print("Content:", message["content"])


# ============================================================
# 8. PRODUCTION PROMPT STRUCTURE
# ============================================================

production_messages = [
    {
        "role": "system",
        "content": (
            "You are an AI assistant for a company's "
            "internal knowledge base. Answer using "
            "the provided context. If the answer is "
            "not available, say you don't know."
        )
    },
    {
        "role": "user",
        "content": (
            "What is the company's leave policy?"
        )
    }
]

print("\n===================================")
print("PRODUCTION PROMPT")
print("===================================")

for message in production_messages:
    print(f"{message['role'].upper()}:")
    print(message["content"])


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n===================================")
print("WEEK 4 - DAY 1 SUMMARY")
print("===================================")

print("""
1. Tokens
   Text is divided into tokens that models process.

2. Attention
   Helps the transformer determine which tokens
   are relevant to each other.

3. Zero-shot
   Task instruction without examples.

4. Few-shot
   Task instruction with a few examples.

5. Reasoning / Chain-of-thought
   Encourages structured problem solving.
   In production, prefer concise explanations
   rather than exposing private chain-of-thought.

6. System role
   Defines high-level behavior and instructions.

7. User role
   Contains the user's request.

8. Assistant role
   Contains the model's response.

Production flow:

System
   ↓
User
   ↓
Model
   ↓
Assistant
""")