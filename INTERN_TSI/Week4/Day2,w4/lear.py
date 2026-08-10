# ============================================================
# WEEK 4 - DAY 2: RAG
# Retrieval → Augment → Generate
# Naive RAG vs Advanced RAG
# ============================================================

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD CSV
# ============================================================

df = pd.read_csv("spam.csv")

print(df.head())
print(df.columns)


# ============================================================
# 2. PREPARE DOCUMENTS
# ============================================================

documents = df["text"].dropna().tolist()


# ============================================================
# 3. CREATE EMBEDDINGS
# ============================================================

model = SentenceTransformer("all-MiniLM-L6-v2")

document_embeddings = model.encode(documents)


# ============================================================
# 4. USER QUERY
# ============================================================

query = input("Enter your question: ")


# ============================================================
# 5. RETRIEVAL
# ============================================================

query_embedding = model.encode([query])

scores = cosine_similarity(
    query_embedding,
    document_embeddings
)[0]

top_indices = scores.argsort()[-3:][::-1]

retrieved_docs = [
    documents[i]
    for i in top_indices
]


print("\nRetrieved Documents:")
for doc in retrieved_docs:
    print("-", doc)


# ============================================================
# 6. AUGMENT
# ============================================================

context = "\n".join(retrieved_docs)

prompt = f"""
Answer the question using the following context.

Context:
{context}

Question:
{query}
"""

print("\nAugmented Prompt:")
print(prompt)


# ============================================================
# 7. GENERATE
# ============================================================

# Send the augmented prompt to your LLM here.

print("\nGenerate:")
print("The augmented prompt is sent to the LLM.")


# ============================================================
# 8. NAIVE RAG
# ============================================================

print("""
Naive RAG:

Query
 ↓
Embedding
 ↓
Similarity Search
 ↓
Top-K Documents
 ↓
Context
 ↓
LLM
 ↓
Answer
""")


# ============================================================
# 9. ADVANCED RAG
# ============================================================

print("""
Advanced RAG:

Query
 ↓
Query Processing
 ↓
Retrieval
 ↓
Filtering / Reranking
 ↓
Relevant Context
 ↓
LLM
 ↓
Answer
""")


# ============================================================
# 10. DIFFERENCE
# ============================================================

print("""
Naive RAG:
Simple retrieval and fixed Top-K results.

Advanced RAG:
Improves retrieval using techniques such as
query rewriting, filtering, hybrid search,
reranking, and better context selection.
""")