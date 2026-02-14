Historical Figures Chatbot - Readme

Overview:
This repository contains `history_chatbot.py`, a Gradio RAG chatbot that:
- Loads `docs/historical_figures.pdf` and indexes it into a FAISS vector DB
- Uses Ollama embeddings and Ollama chat model for responses
- Tracks conversation history in memory and serves a Gradio UI on port 7860

Prerequisites:
- Python 3.10+ (3.11 recommended, but avoid 3.14 due to compatibility issues)
- Git (optional)
- Ollama installed and running

Setup (Windows example):
1. Create and activate a virtual environment
    python -m venv venv
    & venv\Scripts\Activate.ps1   # PowerShell

2. Install Python dependencies
    pip install -r requirements.txt

3. Ensure the PDF exists at `docs/historical_figures.pdf`.
   If not present, place the PDF at that path or update `PDF_PATH` in `history_chatbot.py`.

4. Install Ollama:
   - Download and install Ollama from https://ollama.ai/download/windows
   - After installation, open a terminal and run: `ollama serve` (this starts the Ollama server)

5. Pull the required models:
   - `ollama pull granite-embedding:latest` (for embeddings)
   - `ollama pull llama3` (for the chat model)

6. Create .env file (optional)
Environment variables:
- LANGCHAIN_API_KEY: Your LangChain/LangSmith API key (optional, for tracing)
- LANGCHAIN_PROJECT: Project name (defaults to `HistoricalFiguresChatbot`)
- LANGCHAIN_TRACING_V2: Set to `true` or `false` to enable/disable tracing

You can set env vars in PowerShell like:
    $env:LANGCHAIN_API_KEY = "your_api_key_here"
    $env:LANGCHAIN_TRACING_V2 = "false"

Run the app:
1. From the project root (where `history_chatbot.py` lives):
    python history_chatbot.py

2. The script launches a Gradio UI. Open the printed local URL (http://127.0.0.1:7860 by default) in your browser.

Notes & Troubleshooting:
- PDF not found: The script raises FileNotFoundError if `docs/historical_figures.pdf` is missing.
- Ollama connection errors: Ensure Ollama is running (`ollama serve`) and models are pulled.
- FAISS DB: The vector DB is persisted under `history_chatbot_chroma_db`. Remove that folder to force re-indexing.
- Prompt/input errors (langchain): Ensure the prompt input types are mapping/dict as required by the prompt/runnable.

File references:
- `history_chatbot.py` (main script)
- `requirements.txt` (dependencies)
- `docs/historical_figures.pdf` (content source)
- `history_chatbot_chroma_db/` (FAISS persistence)