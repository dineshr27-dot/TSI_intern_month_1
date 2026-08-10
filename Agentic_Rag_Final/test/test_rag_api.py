from langchain_groq import ChatGroq

from rag.loader import load_documents, split_documents
from rag.retriever import create_vector_store, get_hybrid_retriever


# ============================================================
# 1. Load Documents
# ============================================================

documents = load_documents()
print(f"Pages Loaded: {len(documents)}")


# ============================================================
# 2. Split Documents
# ============================================================

chunks = split_documents(documents)
print(f"Chunks Created: {len(chunks)}")


# ============================================================
# 3. Create Vector Store
# ============================================================

create_vector_store(chunks)

retriever = get_hybrid_retriever(chunks)


# ============================================================
# 4. Load Groq LLM
# ============================================================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=2048
)


# ============================================================
# 5. User Question
# ============================================================

query = input("\nAsk a question from the documents: ").strip()

if not query:
    print("Please enter a question.")
    raise SystemExit


# ============================================================
# 6. Retrieve Top 3 Chunks
# ============================================================

docs = retriever.invoke(query)[:3]

if not docs:
    print("No relevant documents found.")
    raise SystemExit


# ============================================================
# 7. Create Context
# ============================================================

context = "\n\n".join(
    doc.page_content for doc in docs
)


# ============================================================
# 8. Create RAG Prompt
# ============================================================

prompt = f"""
You are a document question-answering assistant.

Answer the question using ONLY the provided document context.

If the answer is not available in the context, say:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""


# ============================================================
# 9. Send Context + Question to LLM
# ============================================================

response = llm.invoke(prompt)


# ============================================================
# 10. Display Answer
# ============================================================

print("\n" + "=" * 60)
print("FINAL ANSWER")
print("=" * 60)

print(response.content)


# ============================================================
# 11. Display Retrieved Sources
# ============================================================

print("\n" + "=" * 60)
print("RETRIEVED SOURCES")
print("=" * 60)

for i, doc in enumerate(docs, start=1):

    source = doc.metadata.get(
        "source",
        "Unknown source"
    )

    print(f"{i}. {source}")