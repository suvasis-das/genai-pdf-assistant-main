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

st.title("🤖 NoteBot AI – Smart AI Assistant")
st.caption("RAG + General AI + Study Modes")

# =======================
# SAFETY
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

    if not text.strip():
        st.error("❌ No readable text found in PDF.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
    st.success("✅ PDF Ready!")

# =======================
# 💬 SINGLE SMART CHAT INPUT
# =======================
user_query = st.chat_input("Ask anything (PDF / General / Study)...")

if user_query:

    # Show user
    with st.chat_message("user"):
        st.markdown(user_query)

    # Savage check
    if detect_unfair_query(user_query):
        answer = savage_reply()

    else:
        try:
            # =======================
            # CASE 1: PDF AVAILABLE → RAG
            # =======================
            if st.session_state.vector_store:

                docs = st.session_state.vector_store.similarity_search(user_query, k=k)
                context = "\n\n".join([d.page_content for d in docs])

                # Study modes
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
Use PDF context first. If insufficient, use general knowledge.

Context:
{context}

Question:
{user_query}
"""

                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    temperature=0.7,
                    max_tokens=1024,
                    top_p=0.9,
                    messages=[
                        {"role": "system", "content": "You are a smart study assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )

                answer = response.choices[0].message.content

            # =======================
            # CASE 2: NO PDF → GENERAL AI
            # =======================
            else:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    temperature=0.7,
                    max_tokens=1024,
                    top_p=0.9,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": user_query}
                    ]
                )

                answer = response.choices[0].message.content

        except Exception as e:
            answer = f"❌ Error: {e}"

    # Show assistant
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Save history
    st.session_state.chat_history.append(("user", user_query))
    st.session_state.chat_history.append(("bot", answer))

# =======================
# CHAT HISTORY
# =======================
for role, msg in st.session_state.chat_history:
    if role == "user":
        with st.chat_message("user"):
            st.markdown(msg)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg)