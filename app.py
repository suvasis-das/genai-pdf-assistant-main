import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os
import random

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from groq import Groq

# =======================
# LOAD ENV
# =======================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ API KEY missing")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# =======================
# UI CONFIG
# =======================
st.set_page_config(page_title="NoteBot AI", page_icon="🤖", layout="wide")

st.title("🤖 NoteBot AI – RAG Based PDF Assistant")
st.caption("Built using Generative AI, RAG & Vector Search")

# =======================
# SAFETY (SAVAGE MODE)
# =======================
def detect_unfair_query(query):
    keywords = [
        "exact question paper",
        "paper leak",
        "leak paper",
        "exact questions",
        "predict exact"
    ]
    return any(k in query.lower() for k in keywords)

def savage_reply():
    replies = [
        "Nice try 😄 Go study properly and come back.",
        "If I had the paper, I'd be giving exams too 😂",
        "No shortcuts here. Learn properly.",
        "I help you study, not cheat 😉"
    ]
    return random.choice(replies)

# =======================
# SIDEBAR
# =======================
with st.sidebar:

    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if files:
        st.success(f"{len(files)} PDF(s) uploaded")

    k = st.slider("Accuracy Control (Top-K)", 1, 10, 5)

    st.markdown("### 📚 Study Mode")
    study_mode = st.selectbox(
        "Choose",
        ["Ask Question", "Summarize PDF", "Important Points", "Notes", "Questions", "Explain Simple"]
    )

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("Reset Chat"):
        st.session_state.chat_history = []
        st.session_state.vector_store = None
        st.rerun()

# =======================
# SESSION STATE
# =======================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# =======================
# PDF PROCESSING
# =======================
if files and st.session_state.vector_store is None:

    text = ""

    for file in files:
        pdf = PdfReader(file)

        for page in pdf.pages:
            try:
                content = page.extract_text()
                if content:
                    text += content
            except:
                pass

    # 🚨 Check if text extracted
    if not text.strip():
        st.error("❌ No readable text found in PDF.\n👉 Try a text-based PDF (not scanned).")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    # 🚨 Check chunks
    if not chunks:
        st.error("❌ Failed to split text into chunks.")
        st.stop()

    # DEBUG (optional)
    st.write(f"📄 Text length: {len(text)}")
    st.write(f"🧩 Chunks created: {len(chunks)}")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)

    except Exception as e:
        st.error(f"❌ Embedding error: {e}")
        st.stop()

    st.success("✅ PDF Ready!")

# =======================
# INPUT
# =======================
query = None

if study_mode == "Ask Question":
    query = st.chat_input("Ask your question...")
else:
    if st.button("Run"):
        query = "RUN"

# =======================
# MAIN LOGIC
# =======================
if (query or study_mode != "Ask Question") and st.session_state.vector_store:

    if study_mode == "Ask Question":

        if detect_unfair_query(query):
            answer = savage_reply()
            st.session_state.chat_history.append(("bot", answer))
            st.markdown(f"**🤖 {answer}**")
            st.stop()

        st.session_state.chat_history.append(("user", query))

    docs = st.session_state.vector_store.similarity_search(
        query if query else "summary",
        k=k
    )

    context = "\n\n".join([d.page_content for d in docs])

    # =======================
    # PROMPTS
    # =======================
    if study_mode == "Summarize PDF":
        prompt = f"Summarize clearly:\n{context}"

    elif study_mode == "Important Points":
        prompt = f"Give key important points:\n{context}"

    elif study_mode == "Notes":
        prompt = f"Make exam notes:\n{context}"

    elif study_mode == "Questions":
        prompt = f"Generate exam questions:\n{context}"

    elif study_mode == "Explain Simple":
        prompt = f"Explain simply:\n{context}"

    else:
        prompt = f"""
Answer ONLY from context.
If not found say "I don't know".

Context:
{context}

Question:
{query}
"""

    # =======================
    # LLM CALL
    # =======================
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content

    except Exception as e:
        answer = f"❌ LLM Error: {e}"

    st.session_state.chat_history.append(("bot", answer))

    st.markdown(f"**🤖 {answer}**")

    st.download_button("Download Answer", answer)

    with st.expander("📚 Source"):
        for d in docs[:2]:
            st.write(d.page_content)

# =======================
# CHAT DISPLAY
# =======================
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"**🧑 {msg}**")
    else:
        st.markdown(f"**🤖 {msg}**")