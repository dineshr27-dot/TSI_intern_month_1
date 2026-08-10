from rag.loader import load_documents, split_documents
from rag.retriever import create_vector_store, get_hybrid_retriever

# Load documents
documents = load_documents()
print(f"Pages Loaded: {len(documents)}")

# Split documents
chunks = split_documents(documents)
print(f"Chunks Created: {len(chunks)}")

# Create ChromaDB
create_vector_store(chunks)
print(" ChromaDB created successfully!")

# Load Hybrid Retriever
retriever = get_hybrid_retriever(chunks)

query =(input("Ask any question in the documnet any : "))
# Retrieve only top 3 chunks
results = retriever.invoke(query)[:3]

print("\nTop 3 Results:\n")

for i, doc in enumerate(results, start=1):
    print(f"Result {i}")
    print("-" * 50)
    print(doc.page_content)
    print()