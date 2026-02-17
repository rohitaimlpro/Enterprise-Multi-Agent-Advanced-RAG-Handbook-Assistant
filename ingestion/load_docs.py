import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


def load_handbook_pdfs(folder_path: str) -> List[Document]:
    """
    Loads all PDFs from folder_path using PyPDFLoader.
    Returns list of LangChain Documents (page-wise).
    Each Document will have metadata: source, page.
    """
    all_docs: List[Document] = []

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {folder_path}. Put your handbook PDFs there."
        )

    for pdf_file in pdf_files:
        full_path = os.path.join(folder_path, pdf_file)

        loader = PyPDFLoader(full_path)
        docs = loader.load()  # page-wise documents

        # add clean source name
        for d in docs:
            d.metadata["handbook_name"] = pdf_file
            # keep page metadata already included by loader
            # d.metadata["page"] exists
        all_docs.extend(docs)

    return all_docs