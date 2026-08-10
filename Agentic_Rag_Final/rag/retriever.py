from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


# ============================================================
# ChromaDB Path
# ============================================================

CHROMA_PATH = "chroma_db_new"


# ============================================================
# Embedding Model
# ============================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# Create Vector Store
# ============================================================

def create_vector_store(chunks):
    """
    Create document embeddings and store them in ChromaDB.
    """

    if not chunks:
        raise ValueError("No document chunks provided.")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH
    )

    return db


# ============================================================
# Create Hybrid Retriever
# ============================================================

def get_hybrid_retriever(chunks):
    """
    Create a hybrid retriever combining:

    1. BM25 keyword retrieval
    2. ChromaDB semantic retrieval

    BM25 weight: 0.4
    Semantic search weight: 0.6
    """

    if not chunks:
        raise ValueError("No document chunks provided.")

    # --------------------------------------------------------
    # Load ChromaDB
    # --------------------------------------------------------

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )

    # --------------------------------------------------------
    # Semantic Retriever
    # --------------------------------------------------------

    dense_retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10,
            "lambda_mult": 0.7
        }
    )

    # --------------------------------------------------------
    # BM25 Keyword Retriever
    # --------------------------------------------------------

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 3

    # --------------------------------------------------------
    # Hybrid Retriever
    # --------------------------------------------------------

    hybrid_retriever = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            dense_retriever
        ],
        weights=[
            0.4,
            0.6
        ]
    )

    return hybrid_retriever