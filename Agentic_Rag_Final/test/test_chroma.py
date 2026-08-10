from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rag.loader import load_documents, split_documents


print("1. Loading documents")

documents = load_documents()
chunks = split_documents(documents)

print(f"Pages: {len(documents)}")
print(f"Chunks: {len(chunks)}")


print("2. Loading embedding model")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("3. Embedding model loaded")


print("4. Creating ChromaDB")

db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="test_chroma_db"
)

print("5. ChromaDB created")


print("6. Searching ChromaDB")

results = db.similarity_search(
    "health",
    k=3
)

print(f"Results found: {len(results)}")

for i, doc in enumerate(results, start=1):
    print(f"\nResult {i}:")
    print(doc.page_content[:300])


print("\nTEST COMPLETED")