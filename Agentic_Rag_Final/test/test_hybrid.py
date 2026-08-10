from rag.loader import load_documents, split_documents
from rag.retriever import create_vector_store, get_hybrid_retriever


print("1. Loading documents")

documents = load_documents()
chunks = split_documents(documents)

print(f"Pages: {len(documents)}")
print(f"Chunks: {len(chunks)}")


print("2. Creating vector store")

create_vector_store(chunks)

print("3. Vector store created")


print("4. Creating hybrid retriever")

retriever = get_hybrid_retriever(chunks)

print("5. Hybrid retriever created")


print("6. Testing hybrid search")

query = input("\nEnter document question: ")

results = retriever.invoke(query)[:3]

print(f"\nResults found: {len(results)}")


for i, doc in enumerate(results, start=1):

    print(f"\nResult {i}")
    print("-" * 50)
    print(doc.page_content[:500])


print("\nHYBRID TEST COMPLETED")