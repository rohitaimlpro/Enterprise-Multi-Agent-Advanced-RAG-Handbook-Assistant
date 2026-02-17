from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def main():
    CHROMA_DIR = "data/vectorstore"
    COLLECTION_NAME = "company_handbooks"

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )

    queries = [
        "notice period",
        "probation period",
        "leave policy",
        "work from home policy",
        "termination rules",
        "employee benefits",
    ]

    for q in queries:
        print("\n" + "=" * 80)
        print(f"🔎 QUERY: {q}")
        print("=" * 80)

        results = vectordb.similarity_search(q, k=5)

        for i, doc in enumerate(results, start=1):
            handbook = doc.metadata.get("handbook_name", "unknown")
            page = doc.metadata.get("page", "N/A")
            chunk_id = doc.metadata.get("chunk_id", "N/A")

            print(f"\n--- Result {i} ---")
            print(f"📘 Handbook: {handbook}")
            print(f"📄 Page: {page} | Chunk: {chunk_id}")
            print(doc.page_content[:600])


if __name__ == "__main__":
    main()