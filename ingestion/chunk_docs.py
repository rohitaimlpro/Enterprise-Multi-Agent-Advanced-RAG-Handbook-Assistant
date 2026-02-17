from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ingestion.clean_text import clean_handbook_text


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 1200,
    chunk_overlap: int = 250
) -> List[Document]:
    """
    Handbook-optimized chunking.
    - chunk_size is in characters (LangChain splitter is char-based).
    - We use bigger chunks because handbook sections are long.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunked_docs: List[Document] = []

    for doc in docs:
        cleaned = clean_handbook_text(doc.page_content)

        if len(cleaned) < 30:
            continue

        # split text into chunks
        chunks = splitter.split_text(cleaned)

        for idx, chunk in enumerate(chunks):
            new_doc = Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,
                    "chunk_id": idx,
                },
            )
            chunked_docs.append(new_doc)

    return chunked_docs