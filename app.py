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
client = Groq(api_key=GROQ_API_KEY)

# =======================
# UI CONFIG
# =======================
st.set_page_config(page_title="NoteBot AI", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0E1117, #1a1c24);
    color: white;
}
.chat-box {
    padding: 14px;
    border-radius: 15px;
    margin-bottom: 12px;
    max-width: 75%;
    font-size: 15px;
}
.user {
    background: linear-gradient(135deg, #1f77b4, #4facfe);
    margin-left: auto;
}
.bot {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
}
.stButton>button {
    border-radius: 10px;
    background: linear-gradient(135deg, #1f77b4, #4facfe);
    color: white;
}
</style>
""", unsafe_allow_html=True)

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
        for f in files:
            st.write(f"📄 {f.name}")

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

# =======================
# SESSION
# =======================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# =======================
# PROCESS PDF
# =======================
if files and st.session_state.vector_store is None:

    text = ""

    for file in files:
        pdf = PdfReader(file)
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)

    st.success("PDF Ready!")

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

        # 🔥 Savage filter
        if detect_unfair_query(query):
            answer = savage_reply()
            st.session_state.chat_history.append(("bot", answer))
            st.markdown(f'<div class="chat-box bot"><b>{answer}</b></div>', unsafe_allow_html=True)
            st.stop()

        st.session_state.chat_history.append(("user", query))

    docs = st.session_state.vector_store.similarity_search(query if query else "summary", k=k)

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
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    st.session_state.chat_history.append(("bot", answer))

    st.markdown(f'<div class="chat-box bot">{answer}</div>', unsafe_allow_html=True)

    st.download_button("Download Answer", answer)

    with st.expander("Source"):
        for d in docs[:2]:
            st.write(d.page_content)

# =======================
# DISPLAY CHAT
# =======================
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f'<div class="chat-box user">{msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-box bot">{msg}</div>', unsafe_allow_html=True)