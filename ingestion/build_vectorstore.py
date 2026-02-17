import os
from dotenv import load_dotenv
from ingestion.load_docs import load_handbook_pdfs
from ingestion.chunk_docs import chunk_documents

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def main():
    load_dotenv()

    RAW_FOLDER = "data/raw_handbooks"
    CHROMA_DIR = "data/vectorstore"
    COLLECTION_NAME = "company_handbooks"

    os.makedirs(CHROMA_DIR, exist_ok=True)

    print("📄 Loading handbook PDFs...")
    docs = load_handbook_pdfs(RAW_FOLDER)
    print(f"✅ Loaded {len(docs)} PDF pages.")

    print("✂️ Chunking documents...")
    chunked_docs = chunk_documents(docs)
    print(f"✅ Created {len(chunked_docs)} chunks.")

    print("🧠 Creating embeddings + storing in Chroma...")

    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

    vectordb = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    vectordb.persist()

    print("✅ Vector DB stored successfully!")
    print(f"📍 Saved at: {CHROMA_DIR}")
    print(f"📦 Collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()