from rag.loader import load_documents, split_documents

documents = load_documents()

print("Pages Loaded:", len(documents))

chunks = split_documents(documents)

print("Chunks:", len(chunks))

print("\nFirst Chunk:\n")

print(chunks[0].page_content)