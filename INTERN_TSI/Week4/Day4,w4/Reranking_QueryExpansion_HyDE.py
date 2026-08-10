# ============================================================
# WEEK 4 - DAY 4
# Re-ranking, Query Expansion, HyDE,
# Chunk Navigation, ReAct, LangGraph
# ============================================================

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD CSV DATA
# ============================================================

CSV_FILE = "ecommerceDataset.csv"
TEXT_COLUMN = "text"

df = pd.read_csv(CSV_FILE)

documents = df[TEXT_COLUMN].dropna().astype(str).tolist()

print("Documents:", len(documents))


# ============================================================
# 2. CREATE EMBEDDINGS
# ============================================================

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(documents)


# ============================================================
# 3. QUERY
# ============================================================

query = input("Enter your query: ")

query_embedding = model.encode([query])


# ============================================================
# 4. INITIAL RETRIEVAL
# ============================================================

scores = cosine_similarity(
    query_embedding,
    embeddings
)[0]

top_indices = scores.argsort()[-5:][::-1]

results = []

for i in top_indices:
    results.append({
        "index": i,
        "text": documents[i],
        "score": scores[i]
    })

print("\nInitial Retrieval:")

for result in results:
    print(
        round(result["score"], 4),
        "->",
        result["text"]
    )


# ============================================================
# 5. RE-RANKING
# ============================================================

reranked_results = sorted(
    results,
    key=lambda x: x["score"],
    reverse=True
)

print("\nRe-ranked Results:")

for result in reranked_results:
    print(
        round(result["score"], 4),
        "->",
        result["text"]
    )


# ============================================================
# 6. QUERY EXPANSION
# ============================================================

expanded_queries = [
    query,
    query + " explanation",
    query + " details",
    query + " information"
]

print("\nExpanded Queries:")

for q in expanded_queries:
    print("-", q)


# ============================================================
# 7. RETRIEVE USING EXPANDED QUERIES
# ============================================================

all_results = []

for expanded_query in expanded_queries:

    q_embedding = model.encode([expanded_query])

    scores = cosine_similarity(
        q_embedding,
        embeddings
    )[0]

    top_indices = scores.argsort()[-3:][::-1]

    for i in top_indices:

        all_results.append({
            "index": i,
            "text": documents[i],
            "score": scores[i]
        })


# Remove duplicate documents
unique_results = {}

for result in all_results:

    index = result["index"]

    if (
        index not in unique_results
        or result["score"] > unique_results[index]["score"]
    ):
        unique_results[index] = result


expanded_results = sorted(
    unique_results.values(),
    key=lambda x: x["score"],
    reverse=True
)


print("\nQuery Expansion Results:")

for result in expanded_results[:5]:
    print(
        round(result["score"], 4),
        "->",
        result["text"]
    )


# ============================================================
# 8. HyDE
# ============================================================

hypothetical_document = (
    "This document should contain information "
    "that directly answers the question: " + query
)

hyde_embedding = model.encode(
    [hypothetical_document]
)

hyde_scores = cosine_similarity(
    hyde_embedding,
    embeddings
)[0]

hyde_indices = hyde_scores.argsort()[-3:][::-1]

print("\nHyDE Results:")

for i in hyde_indices:

    print(
        round(hyde_scores[i], 4),
        "->",
        documents[i]
    )


# ============================================================
# 9. CHUNK NAVIGATION
# ============================================================

df["chunk_id"] = range(len(df))

df["prev_chunk_id"] = df["chunk_id"].shift(1)

df["next_chunk_id"] = df["chunk_id"].shift(-1)

print("\nChunk Navigation:")

print(
    df[
        [
            "chunk_id",
            "prev_chunk_id",
            "next_chunk_id"
        ]
    ].head()
)


# ============================================================
# 10. GET PREVIOUS / NEXT CHUNK
# ============================================================

def get_chunk(chunk_id):

    row = df[df["chunk_id"] == chunk_id]

    if row.empty:
        return None

    return row.iloc[0]


current_chunk_id = 0

current = get_chunk(current_chunk_id)

print("\nCurrent Chunk:")
print(current[TEXT_COLUMN])

if pd.notna(current["prev_chunk_id"]):

    previous = get_chunk(
        int(current["prev_chunk_id"])
    )

    print("\nPrevious Chunk:")
    print(previous[TEXT_COLUMN])

if pd.notna(current["next_chunk_id"]):

    next_chunk = get_chunk(
        int(current["next_chunk_id"])
    )

    print("\nNext Chunk:")
    print(next_chunk[TEXT_COLUMN])


# ============================================================
# 11. REACT LOOP
# ============================================================

print("\nReAct Loop:")

print("Reason → Decide what information is needed")

print("Act → Search the dataset")

react_scores = cosine_similarity(
    query_embedding,
    embeddings
)[0]

best_index = react_scores.argmax()

print("Observe → Retrieved:")
print(documents[best_index])

print("Reason → Retrieved information is relevant")

print("Final Answer → Use retrieved information")


# ============================================================
# 12. LANGGRAPH OVERVIEW
# ============================================================

print("""
LangGraph workflow:

START
  ↓
Query
  ↓
Retrieve
  ↓
Re-rank
  ↓
 ┌───────────────┐
 │ Relevant?     │
 └───────────────┘
      ↓
     YES
      ↓
 Generate
      ↓
     END

If NOT relevant:

Retrieve
   ↓
Query Expansion
   ↓
Retrieve Again
   ↓
Re-rank
   ↓
Generate
   ↓
END
""")


# ============================================================
# SUMMARY
# ============================================================

print("""
============================================================
SUMMARY
============================================================

Re-ranking:
Reorders retrieved results according to relevance.

Query Expansion:
Creates additional versions of the original query.

HyDE:
Creates a hypothetical document and uses it for retrieval.

Chunk Navigation:
Uses previous_chunk_id and next_chunk_id to navigate
through neighboring chunks.

ReAct:
Reason → Act → Observe → Reason → Answer

LangGraph:
Represents the agent workflow as nodes and edges.
============================================================
""")