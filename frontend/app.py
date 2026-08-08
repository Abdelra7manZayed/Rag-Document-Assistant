"""
RAG Document Assistant — Streamlit Frontend
Run with: streamlit run app.py
"""
import requests
import streamlit as st

from api_client import ask_question, check_health

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="centered",
)

# ── session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {"role": "user"|"assistant", "content": str}


# ── header ────────────────────────────────────────────────────────────────────
st.title("📚 RAG Document Assistant")
st.caption("Ask questions — every answer is grounded in your documents.")

# Backend health badge
with st.sidebar:
    st.header("⚙️ Status")
    if check_health():
        st.success("✅ Backend online")
    else:
        st.error("❌ Backend offline — start the FastAPI server first.")
        st.code("cd backend\nuvicorn app.main:app --reload", language="bash")

    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown(
        "1. Your question is embedded and matched against document chunks.\n"
        "2. The top matching chunks are sent to the LLM as context.\n"
        "3. The LLM answers *only* from that context — no hallucination."
    )
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ── chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']}** — chunk `{s['chunk_id']}`")
                    st.caption(s["excerpt"])


# ── input box ─────────────────────────────────────────────────────────────────
question = st.chat_input("Ask something about your documents…")

if question:
    # Show the user's message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Call the backend and stream a spinner
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer…"):
            try:
                result = ask_question(question)
                answer = result["answer"]
                sources = result.get("sources", [])
                model = result.get("model_used", "unknown")

                st.markdown(answer)
                st.caption(f"Model: `{model}`")

                if sources:
                    with st.expander("📎 Sources"):
                        for s in sources:
                            st.markdown(f"**{s['source']}** — chunk `{s['chunk_id']}`")
                            st.caption(s["excerpt"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except requests.ConnectionError:
                err = "❌ Cannot reach the backend. Is the FastAPI server running?"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

            except requests.Timeout:
                err = "⏳ The request timed out. The LLM may be overloaded — try again."
                st.warning(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

            except requests.HTTPError as e:
                err = f"⚠️ Backend error: {e.response.status_code} — {e.response.text}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
