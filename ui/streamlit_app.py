import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Enterprise Handbook RAG",
    page_icon="📘",
    layout="wide"
)

st.title("📘 Enterprise Handbook RAG Assistant")
st.caption("Advanced RAG + LangGraph Supervisor + Hybrid Retrieval + Memory + Verification")

# thread_id persists per session
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat" not in st.session_state:
    st.session_state.chat = []


with st.sidebar:
    st.subheader("⚙️ Settings")
    st.write("Thread ID:")
    st.code(st.session_state.thread_id)

    if st.button("🧹 Reset Conversation"):
        st.session_state.chat = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.write("API URL:")
    st.code(API_URL)


query = st.chat_input("Ask about company policies... (leave, notice period, WFH, benefits etc.)")


# Render history
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if query:
    # store user message
    st.session_state.chat.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            payload = {
                "query": query,
                "thread_id": st.session_state.thread_id
            }

            try:
                r = requests.post(
                    f"{API_URL}/chat",
                    json=payload,
                    timeout=60
                )
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Is FastAPI running on port 8000?")
                st.stop()
            except requests.exceptions.Timeout:
                st.error("⏳ API request timed out. Try again.")
                st.stop()

            # If API returns error (500, 404, etc.)
            if r.status_code != 200:
                st.error(f"❌ API Error: {r.status_code}")
                st.code(r.text)
                st.stop()

            # If response is not JSON
            try:
                data = r.json()
            except Exception:
                st.error("❌ API did not return JSON.")
                st.code(r.text)
                st.stop()

            answer = data.get("answer", "")
            confidence = data.get("confidence", 0)
            grounded = data.get("is_grounded", False)
            issues = data.get("issues", [])
            sources = data.get("sources", [])
            logs = data.get("stream_log", [])

            st.markdown(answer)

            st.divider()

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("✅ Verification")
                st.write(f"**Confidence:** {confidence}/100")
                st.write(f"**Grounded:** {grounded}")

                if issues:
                    st.write("**Issues:**")
                    for it in issues:
                        st.write(f"- {it}")

            with col2:
                st.subheader("📌 Sources")
                if sources:
                    for s in sources:
                        st.write(f"- **[{s.get('id', '?')}]** {s.get('text', '')}")
                else:
                    st.write("No citations extracted.")

            st.divider()
            st.subheader("🧾 Internal Agent Logs")
            if logs:
                for l in logs:
                    st.write(f"- {l}")
            else:
                st.write("No logs found.")

    # store assistant message
    st.session_state.chat.append({"role": "assistant", "content": answer})