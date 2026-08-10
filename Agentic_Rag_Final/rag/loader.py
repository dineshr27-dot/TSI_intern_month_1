import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCUMENT_PATH = "documents"


def load_documents():
    """
    Load all PDF documents from the documents folder.
    """

    documents = []

    for file in os.listdir(DOCUMENT_PATH):

        if file.endswith(".pdf"):

            loader = PyPDFLoader(os.path.join(DOCUMENT_PATH, file))

            documents.extend(loader.load())

    return documents


def split_documents(documents):
    """
    Split documents into chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    return chunks