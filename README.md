NoteBot AI — GenAI PDF Study Assistant
NoteBot AI is a Python-based study assistant that lets users upload PDF documents and ask questions about their contents using natural language. It uses a Retrieval-Augmented Generation (RAG) workflow: the application extracts text from PDFs, splits it into smaller chunks, creates vector embeddings, retrieves relevant chunks for a question, and sends the retrieved context to a Large Language Model (LLM) to generate an answer.

The app also supports general AI chat when no PDF has been processed, voice input, and text-to-speech playback of the latest response.
Features
- PDF upload: Upload one or multiple text-based PDF files through the Streamlit interface.
- Ask questions about documents: Ask natural-language questions and receive answers generated using relevant retrieved PDF text.
- Retrieval-Augmented Generation: Uses text chunking, Hugging Face sentence embeddings, and FAISS similarity search to retrieve context for each query.
- Adjustable retrieval: Change the Top-K setting (1–10) to control how many matching text chunks are retrieved.
- General AI chat: Use the assistant for general questions when no PDF vector store is available.
- Voice input: Submit a question through a microphone using SpeechRecognition.
- Read answers aloud: Convert the latest answer to speech with gTTS.
- Chat history: Keep the conversation visible during the active Streamlit session.
- Response display effect: Display the generated answer with a word-by-word animation.
- Basic keyword guardrail: Detects a small set of exam-misconduct-related phrases and returns a fixed response.
Tech Stack
Area	Technology
Programming language	Python
User interface	Streamlit
PDF text extraction	PyPDF2
Text chunking	LangChain RecursiveCharacterTextSplitter
Embedding model	Hugging Face sentence-transformers/all-MiniLM-L6-v2
Vector store / similarity search	FAISS (faiss-cpu)
LLM provider	Groq API
Configured chat model	llama-3.1-8b-instant
Voice input	SpeechRecognition (Google Web Speech recognition)
Text-to-speech	gTTS
Environment configuration	python-dotenv / Streamlit Secrets


How It Works
1. Upload PDFs: The user uploads one or more PDF files in the Streamlit sidebar.
2. Extract text: PyPDF2 reads the PDF pages and extracts selectable text.
3. Split into chunks: RecursiveCharacterTextSplitter divides the extracted text into chunks of 500 characters with 100 characters of overlap.
4. Create embeddings: Hugging Face's all-MiniLM-L6-v2 model converts each chunk into a numerical embedding.
5. Build the vector store: FAISS stores the embeddings and supports similarity search.
6. Retrieve context: When the user asks a question, the app retrieves the Top-K most similar chunks from the vector store.
7. Generate an answer: The retrieved text is included in a prompt sent to the configured Groq chat model. The model generates a response using that context.
8. Display or speak the answer: Streamlit displays the response, and the user can request audio playback of the latest answer.
PDF upload
    |
    v
Text extraction (PyPDF2)
    |
    v
Text chunking (500 characters, 100 overlap)
    |
    v
Embeddings (Hugging Face: all-MiniLM-L6-v2)
    |
    v
FAISS vector store
    |
    v
User question
    |
    v
Top-K similarity search
    |
    v
Retrieved context + question
    |
    v
Groq LLM (llama-3.1-8b-instant)
    |
    v
Generated answer in Streamlit
    |
    +----> Optional text-to-speech (gTTS)
Getting Started
Prerequisites
- Python 3.10 or a compatible Python version
- A Groq API key
- Internet access for the Groq API, model downloads, and online speech services
- A working microphone for voice input
1. Clone the repository
git clone https://github.com/suvasis-das/genai-pdf-assistant-main.git
cd genai-pdf-assistant-main
2. Create and activate a virtual environment
Windows (PowerShell / Command Prompt):
python -m venv .venv
.venv\Scripts\activate
macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
4. Configure the Groq API key
Create a file named .env in the project root and add your API key:
GROQ_API_KEY=your_groq_api_key_here
Keep this key private. Do not commit .env or paste the API key into source code. The repository's .gitignore excludes .env.

For deployment on Streamlit Community Cloud, add GROQ_API_KEY through the app's Secrets settings instead of committing a secrets file.
5. Run the application
streamlit run app.py
Streamlit will provide a local URL in the terminal, usually http://localhost:8501.
Using the Application
1. Start the application with the command above.
2. Upload one or more text-based PDF files using the sidebar.
3. Adjust the Top-K retrieval value if needed.
4. Enter a question in the chat input, or use the voice-input option.
5. Review the generated response. Use Speak Answer to hear the latest response.
6. To ask general questions without document context, use the assistant when no PDF vector store is loaded.
Project Structure
genai-pdf-assistant-main/
├── app.py              # Streamlit interface and application logic
├── requirements.txt    # Python dependencies
├── .gitignore          # Excludes environment files and local artifacts
└── README.md           # Project documentation
Configuration
- Groq API key: Set GROQ_API_KEY in .env locally or in Streamlit Secrets when deployed.
- Retrieval count: The Top-K slider controls the number of similar chunks retrieved for a query (1–10, default 5).
- Embedding model: The application currently uses sentence-transformers/all-MiniLM-L6-v2.
- Chunk settings: Text is split into 500-character chunks with 100-character overlap in the current implementation.
Current Limitations
- The current PDF extraction flow reads selectable text. Scanned PDFs that contain only page images are not processed with OCR.
- Voice input depends on microphone access and SpeechRecognition's online recognition service. Some systems may require additional audio dependencies or configuration.
- Text-to-speech uses gTTS and requires an internet connection.
- Chat history and the FAISS vector store are held in Streamlit session state; they are not saved as a permanent user database.
- The current sidebar includes a Study Mode selector, but mode-specific prompt behavior is not yet connected to the answer-generation logic.
- The displayed word-by-word response effect is a UI animation after the model response is generated; it is not token streaming from the LLM.
- The current implementation does not display page-level citations or clickable source references for retrieved passages.
Security Notes
- Never commit your Groq API key, .env file, or other secrets.
- Treat generated answers as AI-assisted study material and verify important information against the original document.
- The keyword-based guardrail is a basic demonstration feature, not a complete safety or academic-integrity system.
Future Improvements
- Connect the Study Mode selector to dedicated prompts for summarization, notes, simplified explanations, and question generation.
- Display source excerpts and PDF page references alongside answers.
- Add OCR support for scanned PDFs.
- Improve document management so users can add, remove, and re-index PDFs during a session.
- Add persistent chat history and document sessions.
- Improve error handling for empty, damaged, or image-only PDF files.
- Add automated tests and deployment instructions for a hosted version.
Author
Suvasis Das
B.Tech, Computer Science and Engineering
GitHub: suvasis-das
This project was built as a practical exploration of Generative AI, Retrieval-Augmented Generation, vector search, and interactive document-based applications.
