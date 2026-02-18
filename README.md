# 📘 Enterprise Handbook RAG Assistant (LangGraph + Hybrid Retrieval + Verification)

An **Enterprise Employee Handbook Q&A Assistant** that answers employee policy questions (leave, notice period, probation, benefits, termination, WFH etc.) using **Retrieval-Augmented Generation (RAG)**.

This project uses:

- ✅ **Hybrid Retrieval** (Dense + BM25)
- ✅ **Reranking** (Cross-Encoder)
- ✅ **Multi-hop retrieval** (optional)
- ✅ **LangGraph Supervisor** pipeline
- ✅ **Conversation Memory** (SQLite checkpoints)
- ✅ **Grounding Verification** (local)
- ✅ **FastAPI backend**
- ✅ **Streamlit UI**
- ✅ **Gemini API** (LLM generation)

---

## 🚀 Features

### 🔎 Advanced Retrieval Pipeline

- Dense similarity search using **Chroma + SentenceTransformer**
- Keyword search using **BM25**
- Merged + deduplicated results

### 📌 Reranking

Uses **cross-encoder/ms-marco-MiniLM-L-6-v2** to rerank the retrieved chunks and keep the most relevant ones.

### 🧠 Answer Generation

Uses **Gemini (gemini-2.5-flash)** to generate final answers strictly from retrieved context.

### ✅ Grounding Verification

A local verifier checks:

- similarity between answer and context
- missing sources section
- confidence score (0–100)

### 🧾 Citations

Answers contain a **Sources:** section with citations like:

```txt
[1] Employee-Handbook.pdf (page 17, chunk 0)
[2] HR-Handbook.pdf (page 46, chunk 0)
```

### 💾 Memory

LangGraph uses SQLite checkpointing to store thread state and allow conversation continuity.

### 🌐 UI

A clean Streamlit chat UI with:

- answer output
- verification confidence
- sources
- internal agent logs

---

## 🏗️ Project Architecture

```bash
enterprise_handbook_rag/
│
├── agents/
│   ├── langgraph_supervisor.py
│   ├── nodes.py
│   ├── state.py
│   ├── retrieval_agent.py
│   ├── reranker_agent.py
│   ├── multihop_agent.py
│   ├── compressor_agent.py
│   ├── query_understanding_agent.py
│   ├── query_rewrite_agent.py
│   ├── answer_agent.py
│   ├── action_agent.py
│   ├── verifier_agent.py
│   ├── handbook_filter.py
│   └── streaming_agent.py
│
├── ingestion/
│   ├── build_vectorstore.py
│   ├── chunk_docs.py
│   ├── clean_text.py
│   └── pdf_loader.py
│
├── memory/
│   ├── checkpoints.sqlite
│   └── conversation_memory.py
│
├── api/
│   └── app.py
│
├── ui/
│   └── streamlit_app.py
│
├── evaluation/
│   └── test_langgraph.py
│
├── data/
│   └── vectorstore/
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🧠 LangGraph Flow (Supervisor)

The LangGraph pipeline runs:

1. Understand query (intent + action detection)
2. Rewrite query (FLAN-T5 local)
3. Retrieve (Hybrid)
4. Multi-hop retrieval (optional)
5. Rerank (Cross-encoder)
6. Compress context (sentence selection)
7. Answer generation (Gemini)
8. Verify grounding (local)
9. Retry if confidence is weak
10. Optional action agent (email/checklist output)

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone <your-repo-url>
cd enterprise_handbook_rag
```

---

### 2️⃣ Create Virtual Environment (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

---

### 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Add Environment Variables

Create `.env` file:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

You can also optionally add a HuggingFace token to avoid rate limits:

```env
HF_TOKEN=your_huggingface_token_here
```

---

## 📥 Ingestion (Build Vector Store)

Place handbook PDFs inside your ingestion folder (or update the path in the ingestion script). Then run:

```bash
python -m ingestion.build_vectorstore
```

This creates:

```bash
data/vectorstore/
```

using ChromaDB.

---

## ▶️ Run Evaluation Test

Run the LangGraph test script:

```bash
python -m evaluation.test_langgraph
```

---

## 🌐 Run Backend API (FastAPI)

Start the API server:

```bash
uvicorn api.app:app --reload --port 8000
```

API will run at:

```
http://127.0.0.1:8000
```

---

## 💬 Run Streamlit UI

In a new terminal:

```bash
streamlit run ui/streamlit_app.py
```

Streamlit runs at:

```
http://localhost:8501
```

---

## 🧪 Sample Queries to Test

Try questions like:

- **What is the notice period and what happens if I don't serve it fully?**
- **Explain probation period policy**
- **What is the leave policy for sick leave?**
- **Is work from home allowed?**
- **What happens during termination for misconduct?**
- **Write an email to HR requesting casual leave**

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **LangGraph**
- **LangChain**
- **ChromaDB**
- **Sentence Transformers**
- **BM25 (rank-bm25)**
- **Cross-Encoder Reranker**
- **FastAPI**
- **Streamlit**
- **Gemini API**

---

## ⚠️ Known Issues / Limitations

### Gemini Free Tier Quota

Gemini free tier has request limits (`429 RESOURCE_EXHAUSTED`).

**Solution options:**

- Wait for quota reset
- Add billing
- Add fallback local generation mode (recommended)
- Switch to a free HuggingFace inference model

### Multi-Handbook Conflicts

If multiple handbooks contain similar policies, results may mix.

**Fix:** Enable handbook filtering strictly by primary handbook.

---

## 📌 Future Improvements

- Add local answer fallback when Gemini quota is exceeded
- Add proper entity extraction (department, grade, role)
- Add PDF export of answers
- Add admin UI to upload new handbooks
- Add authentication + deployment

---

## 👨‍💻 Author

Built as an end-to-end **Advanced RAG + LangGraph Supervisor** project for enterprise handbook Q&A.

---

## ⭐ If You Like This Project

Give it a ⭐ on GitHub and feel free to fork it.
