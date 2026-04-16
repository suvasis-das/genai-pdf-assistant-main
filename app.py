# import streamlit as st
# from PyPDF2 import PdfReader
# from dotenv import load_dotenv
# import os
# import random

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings

# from groq import Groq

# # =======================
# # LOAD ENV
# # =======================
# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

# if not GROQ_API_KEY:
#     st.error("❌ GROQ API KEY missing")
#     st.stop()

# client = Groq(api_key=GROQ_API_KEY)

# # =======================
# # UI CONFIG
# # =======================
# st.set_page_config(page_title="NoteBot AI", page_icon="🤖", layout="wide")

# st.title("🤖 NoteBot AI – Smart AI Assistant")
# st.caption("RAG + General AI + Study Modes")

# # =======================
# # SAFETY
# # =======================
# def detect_unfair_query(query):
#     keywords = [
#         "exact question paper",
#         "paper leak",
#         "leak paper",
#         "exact questions",
#         "predict exact"
#     ]
#     return any(k in query.lower() for k in keywords)

# def savage_reply():
#     replies = [
#         "Nice try 😄 Go study properly and come back.",
#         "If I had the paper, I'd be giving exams too 😂",
#         "No shortcuts here. Learn properly.",
#         "I help you study, not cheat 😉"
#     ]
#     return random.choice(replies)

# # =======================
# # SIDEBAR
# # =======================
# with st.sidebar:

#     files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

#     if files:
#         st.success(f"{len(files)} PDF(s) uploaded")

#     k = st.slider("Accuracy Control (Top-K)", 1, 10, 5)

#     st.markdown("### 📚 Study Mode")
#     study_mode = st.selectbox(
#         "Choose",
#         ["Ask Question", "Summarize PDF", "Important Points", "Notes", "Questions", "Explain Simple"]
#     )

#     if st.button("Clear Chat"):
#         st.session_state.chat_history = []
#         st.rerun()

#     if st.button("Reset Chat"):
#         st.session_state.chat_history = []
#         st.session_state.vector_store = None
#         st.rerun()

# # =======================
# # SESSION STATE
# # =======================
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# if "vector_store" not in st.session_state:
#     st.session_state.vector_store = None

# # =======================
# # PDF PROCESSING
# # =======================
# if files and st.session_state.vector_store is None:

#     text = ""

#     for file in files:
#         pdf = PdfReader(file)
#         for page in pdf.pages:
#             try:
#                 content = page.extract_text()
#                 if content:
#                     text += content
#             except:
#                 pass

#     if not text.strip():
#         st.error("❌ No readable text found in PDF.")
#         st.stop()

#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#     chunks = splitter.split_text(text)

#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )

#     st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
#     st.success("✅ PDF Ready!")

# # =======================
# # 💬 SINGLE CHAT INPUT
# # =======================
# user_query = st.chat_input("Ask anything (PDF / General / Study)...")

# if user_query:

#     # Save user message FIRST
#     st.session_state.chat_history.append(("user", user_query))

#     if detect_unfair_query(user_query):
#         answer = savage_reply()

#     else:
#         try:
#             # CASE 1: PDF AVAILABLE
#             if st.session_state.vector_store:

#                 docs = st.session_state.vector_store.similarity_search(user_query, k=k)
#                 context = "\n\n".join([d.page_content for d in docs])

#                 if study_mode == "Summarize PDF":
#                     prompt = f"Summarize clearly:\n{context}"

#                 elif study_mode == "Important Points":
#                     prompt = f"Give key important points:\n{context}"

#                 elif study_mode == "Notes":
#                     prompt = f"Make exam notes:\n{context}"

#                 elif study_mode == "Questions":
#                     prompt = f"Generate exam questions:\n{context}"

#                 elif study_mode == "Explain Simple":
#                     prompt = f"Explain simply:\n{context}"

#                 else:
#                     prompt = f"""
# Use PDF context first. If insufficient, use general knowledge.

# Context:
# {context}

# Question:
# {user_query}
# """

#                 response = client.chat.completions.create(
#                     model="llama-3.1-8b-instant",
#                     temperature=0.7,
#                     max_tokens=1024,
#                     top_p=0.9,
#                     messages=[
#                         {"role": "system", "content": "You are a smart study assistant."},
#                         {"role": "user", "content": prompt}
#                     ]
#                 )

#                 answer = response.choices[0].message.content

#             # CASE 2: NO PDF
#             else:
#                 response = client.chat.completions.create(
#                     model="llama-3.1-8b-instant",
#                     temperature=0.7,
#                     max_tokens=1024,
#                     top_p=0.9,
#                     messages=[
#                         {"role": "system", "content": "You are a helpful AI assistant."},
#                         {"role": "user", "content": user_query}
#                     ]
#                 )

#                 answer = response.choices[0].message.content

#         except Exception as e:
#             answer = f"❌ Error: {e}"

#     # Save bot response
#     st.session_state.chat_history.append(("bot", answer))

#     # Rerun to prevent duplicate rendering
#     st.rerun()

# # =======================
# # CHAT DISPLAY (ONLY HERE)
# # =======================
# for role, msg in st.session_state.chat_history:
#     if role == "user":
#         with st.chat_message("user"):
#             st.markdown(msg)
#     else:
#         with st.chat_message("assistant"):
#             st.markdown(msg)


import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os
import random
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from groq import Groq

# Voice
import speech_recognition as sr
from gtts import gTTS

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
st.caption("RAG + Voice + Streaming")

# =======================
# SAFETY
# =======================
def detect_unfair_query(query):
    keywords = ["paper leak", "exact questions", "predict exact"]
    return any(k in query.lower() for k in keywords)

def savage_reply():
    return random.choice([
        "Nice try 😄 Go study properly.",
        "No shortcuts here 😉",
    ])

# =======================
# STREAMING FUNCTION
# =======================
def stream_text(text):
    placeholder = st.empty()
    output = ""

    for word in text.split():
        output += word + " "
        placeholder.markdown(output)
        time.sleep(0.03)

# =======================
# VOICE INPUT
# =======================
def get_voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening...")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        return text
    except:
        return None

# =======================
# VOICE OUTPUT
# =======================
def speak(text):
    tts = gTTS(text)
    tts.save("voice.mp3")
    audio_file = open("voice.mp3", "rb")
    st.audio(audio_file.read(), format="audio/mp3")

# =======================
# SIDEBAR
# =======================
with st.sidebar:
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    k = st.slider("Top-K", 1, 10, 5)

    study_mode = st.selectbox("Mode", [
        "Ask Question", "Summarize PDF", "Notes", "Explain Simple"
    ])

    if st.button("🎤 Use Voice Input"):
        voice_text = get_voice_input()
        if voice_text:
            st.session_state.voice_query = voice_text

# =======================
# SESSION STATE
# =======================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "voice_query" not in st.session_state:
    st.session_state.voice_query = None

# =======================
# PDF PROCESSING
# =======================
if files and st.session_state.vector_store is None:

    text = ""
    for file in files:
        pdf = PdfReader(file)
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
    st.success("✅ PDF Ready!")

# =======================
# INPUT (TEXT OR VOICE)
# =======================
user_query = st.chat_input("Ask anything...")

if st.session_state.voice_query:
    user_query = st.session_state.voice_query
    st.session_state.voice_query = None

# =======================
# MAIN LOGIC
# =======================
if user_query:

    st.session_state.chat_history.append(("user", user_query))

    if detect_unfair_query(user_query):
        answer = savage_reply()

    else:
        try:
            # PDF MODE
            if st.session_state.vector_store:

                docs = st.session_state.vector_store.similarity_search(user_query, k=k)
                context = "\n\n".join([d.page_content for d in docs])

                prompt = f"""
Use PDF context first, else general knowledge.

Context:
{context}

Question:
{user_query}
"""

            else:
                prompt = user_query

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=1024,
                top_p=0.9,
                messages=[
                    {"role": "system", "content": "Helpful AI assistant"},
                    {"role": "user", "content": prompt}
                ]
            )

            answer = response.choices[0].message.content

        except Exception as e:
            answer = f"❌ Error: {e}"

    st.session_state.chat_history.append(("bot", answer))
    st.session_state.last_answer = answer

    st.rerun()

# =======================
# DISPLAY CHAT
# =======================
for i, (role, msg) in enumerate(st.session_state.chat_history):

    with st.chat_message("user" if role == "user" else "assistant"):

        if role == "bot" and i == len(st.session_state.chat_history) - 1:
            stream_text(msg)  # 🔥 streaming only latest response
        else:
            st.markdown(msg)

# =======================
# 🔊 SPEAK BUTTON
# =======================
if "last_answer" in st.session_state:
    if st.button("🔊 Speak Answer"):
        speak(st.session_state.last_answer)