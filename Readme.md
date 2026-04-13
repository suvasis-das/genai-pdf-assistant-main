# 🤖 NoteBot AI — Intelligent PDF Study Assistant (RAG-Based)

A **Generative AI-powered web application** that allows users to upload PDFs and interact with them through natural language queries.

Built using **Retrieval-Augmented Generation (RAG)**, the system retrieves relevant document context and generates accurate, explainable answers using a Large Language Model (LLM).

---

## 📌 Overview

NoteBot AI transforms static PDFs into an **interactive study assistant**.

Instead of manually reading long documents, users can:
- Ask questions
- Generate summaries
- Extract key points
- Create notes and questions

This system combines **semantic search (FAISS)** with **LLM-based reasoning (GROQ)** to deliver fast and context-aware responses.

---

## ✨ Features

### 📄 Core Features
- Upload single or multiple PDF documents
- Ask questions in natural language
- Context-aware answers using RAG
- Fast inference using GROQ LLM
- Semantic search via FAISS vector database

---

### 📚 Study Assistant Features
- 📑 **Summarize PDF** — Quick revision summaries
- 📌 **Important Points Extraction** — Key highlights
- 📝 **Automatic Notes Generation** — Exam-ready notes
- ❓ **Question Generation** — Possible exam questions
- 🧒 **Explain Like I'm Beginner** — Simplified explanations

---

### 🎨 UI/UX Features
- Clean chat-based interface (Streamlit)
- Styled chat bubbles (user vs bot)
- Sidebar controls (study modes + accuracy)
- Download generated answers
- View source context (transparency)

---

### 🛡️ Smart Guardrails
- Detects unfair queries (e.g., "paper leak", "exact questions")
- Responds with ethical, academic guidance
- Prevents misuse of the system

---

## 🧠 How It Works (RAG Pipeline)


PDF Upload
↓
Text Extraction (PyPDF2)
↓
Chunking (Text Splitter)
↓
Embeddings (HuggingFace)
↓
Vector Storage (FAISS)
↓
User Query
↓
Similarity Search (Top-K Retrieval)
↓
LLM Processing (GROQ)
↓
Final Context-Aware Answer


---

## 🧰 Tech Stack

| Component | Technology |
|----------|----------|
| Language | Python 🐍 |
| UI | Streamlit 🎨 |
| LLM | GROQ API ⚡ |
| Embeddings | HuggingFace 🤖 |
| Vector DB | FAISS 📊 |
| PDF Processing | PyPDF2 📄 |

---

## 📁 Project Structure

```
genai-pdf-assistant/
│
├── app.py # Main application (RAG + UI)
├── requirements.txt # Dependencies
├── .env # API keys (not pushed to GitHub)
├── .streamlit/
│ └── config.toml # UI configuration
├── README.md # Project documentation

```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Suupratik/genai-pdf-assistant.git
cd genai-pdf-assistant
2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Add API Key

Create .env file:

GROQ_API_KEY=your_api_key_here
5️⃣ Run Application
streamlit run app.py
☁️ Deployment

Deployed using Streamlit Community Cloud

Steps:
Push code to GitHub
Connect repository to Streamlit Cloud
Add secret:
GROQ_API_KEY = "your_api_key"
Deploy 🚀
🧠 Key Concepts Used
Retrieval-Augmented Generation (RAG)
Large Language Models (LLMs)
Prompt Engineering
Semantic Search
Vector Databases (FAISS)
Embeddings
🔮 Future Improvements
📄 Multi-document comparison
🧠 Persistent chat memory
🔍 Highlight answers inside PDF
📊 Advanced summarization modes
🌐 Multilingual support
📱 Mobile UI optimization
📌 Author

👨‍💻 Supratik Mitra
B.Tech CSE (AI & ML)

⭐ Support

If you found this project useful:

👉 Give it a ⭐ on GitHub
👉 Share it with others

🎯 Final Note

This project demonstrates how Generative AI + Retrieval Systems can transform static documents into intelligent, interactive knowledge systems.


---

# 🔥 WHAT IMPROVED

- More **professional language**
- Added **study features (important!)**
- Added **guardrails (very impressive)**
- Clean **tech + pipeline explanation**
- Proper **deployment section**
- Strong **viva impression**

---