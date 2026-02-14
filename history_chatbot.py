"""
Historical Figures Chatbot with LangChain RAG and LangSmith Tracing.

This chatbot uses:
- PDF ingestion with PyPDFLoader
- CharacterTextSplitter for chunking
- FAISS vector store with OllamaEmbeddings
- Ollama LLM (llama3 or mistral)
- LangSmith for tracing
- Gradio for UI
- InMemoryChatMessageHistory for conversation tracking
"""

import os
from dotenv import load_dotenv
from pathlib import Path
import gradio as gr
load_dotenv()

# LangSmith configuration
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "HistoricalFiguresChatbot")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# CONFIG
PDF_PATH = "./docs/historical_figures.pdf"
CHROMA_DIR = "./history_chatbot_chroma_db"

EMBED_MODEL = "granite-embedding:latest"
LLM_MODEL = "llama3"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

class HistoricalFiguresChatbot:
    """Historical Figures Chatbot with RAG pipeline."""
    
    def __init__(self, pdf_path: str = PDF_PATH):
        """
        Initialize the chatbot with PDF documents and vector store.
        
        Args:
            pdf_path: Path to the PDF file containing historical figures information
        """
        print("Initializing HistoryBot...")
        self.pdf_path = pdf_path
        self.vector_store = None
        self.chat_history = InMemoryChatMessageHistory()
        
        # Initialize the RAG pipeline
        self._load_and_index_documents()
        self.qa_chain = self.qa_chain_invoke
        print("Initialization Complete.")
    
    def _load_and_index_documents(self):
        """Load PDF and create vector store with embeddings."""
        print("Loading PDF documents...")
        
        # Load PDF using PyPDFLoader
        if not Path(self.pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found at {self.pdf_path}")
        
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from PDF")
        
        # Split documents using CharacterTextSplitter
        print("Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        
        # Initialize embeddings with OllamaEmbeddings
        print("Initializing embeddings...")
        embeddings = OllamaEmbeddings(
            model=EMBED_MODEL
        )
        
        # Create or load FAISS vector store
        print("Creating vector store...")
        if Path(CHROMA_DIR).exists():
            print("Loading existing vector store...")
            self.vector_store = FAISS.load_local(CHROMA_DIR, embeddings, allow_dangerous_deserialization=True)
        else:
            self.vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings
            )
            self.vector_store.save_local(CHROMA_DIR)
        print("Vector store initialized successfully")

    def qa_chain_invoke(self, query: str):
        """Invoke the QA chain with the given query."""
        # Define custom prompt template
        prompt_template = """You are HistoryBot, an expert on historical figures. 
        Use the following context about historical figures to answer the user's question.
        If the answer is not in the context, say "I don't have information about that in my knowledge base."

        Context:{context}

        Question: {question}

        Answer: """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        # Initialize Ollama LLM
        llm = ChatOllama(model=LLM_MODEL, temperature=0.2)
        # Retrieve relevant documents
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        relevant_docs = retriever._get_relevant_documents(query, run_manager=None)

        # Combine context from relevant documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Create final prompt with context and question
        final_prompt = prompt.invoke({"context": context, "question": query})
        
        # Get response from LLM
        result = llm.invoke(final_prompt, config=None)
        
        return result.content

    def chat(self, user_message: str) -> str:
        """
        Process user message and return chatbot response.
        
        Args:
            user_message: User's question or statement
            
        Returns:
            Chatbot's response
        """
        if not user_message.strip():
            return "Please enter a question about historical figures."
        
        try:
            # Get response text from QA chain
            response_text = self.qa_chain(user_message)

            # Ensure chat history exists
            if self.chat_history is None:
                self.chat_history = InMemoryChatMessageHistory()

            # Prefer history API methods; fall back to safe appends
            try:
                # Most InMemoryChatMessageHistory implementations support these helpers
                self.chat_history.add_user_message(user_message)
                self.chat_history.add_ai_message(response_text)
            except Exception:
                # Try appending message objects
                try:
                    self.chat_history.add_message(HumanMessage(content=user_message))
                    self.chat_history.add_message(AIMessage(content=response_text))
                except Exception:
                    # Final fallback: keep a simple list entry
                    if isinstance(self.chat_history, list):
                        self.chat_history.append({"human": user_message, "ai": response_text})

            return response_text

        except Exception as e:
            error_message = f"Error processing your question: {str(e)}"
            print(f"Error: {error_message}")
            return error_message
    
    def get_chat_history(self) -> str:
        """Get formatted chat history."""
        if not self.chat_history:
            return "No conversation history yet."

        history_text = ""

        # Newer InMemoryChatMessageHistory stores a `.messages` list
        if hasattr(self.chat_history, "messages"):
            for msg in self.chat_history.messages:
                role = getattr(msg, "type", None) or getattr(msg, "role", "")
                content = getattr(msg, "content", str(msg))
                history_text += f"{role}: {content}\n\n"
            return history_text.strip()

        # If it's a plain list of entries
        if isinstance(self.chat_history, list):
            for entry in self.chat_history:
                history_text += f"{entry}\n\n"
            return history_text.strip()

        # Fallback
        return str(self.chat_history)
    
    def clear_history(self):
        """Clear chat history."""
        self.chat_history = InMemoryChatMessageHistory()
        return "Chat history cleared."


def create_gradio_interface():
    """Create and launch Gradio interface."""
    
    # Initialize chatbot
    chatbot = HistoricalFiguresChatbot()
    
    # Create Gradio interface
    with gr.Blocks(title="Historical Figures Chatbot") as interface:
        gr.Markdown("# 📚 Historical Figures Chatbot")
        gr.Markdown("Hello, I am HistoryBot, your expert on historical figures. How can I assist you today?")
        
        with gr.Row():
            with gr.Column(scale=7):
                user_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask me about historical figures...",
                    lines=2,
                    interactive=True
                )
            with gr.Column(scale=1):
                submit_btn = gr.Button("Submit", variant="primary")
        
        with gr.Row():
            response_output = gr.Textbox(
                label="HistoryBot Response",
                interactive=False,
                lines=5
            )
        
        with gr.Row():
            with gr.Column(scale=2):
                clear_btn = gr.Button("Clear History", variant="secondary")
            with gr.Column(scale=10):
                history_output = gr.Textbox(
                    label="Conversation History",
                    interactive=False,
                    lines=8
                )
        
        # Define button interactions
        def process_input(user_message):
            """Process user input and update outputs."""
            response = chatbot.chat(user_message)
            history = chatbot.get_chat_history()
            return response, history
        
        def clear_chat():
            """Clear chat history."""
            chatbot.clear_history()
            return "", ""
        
        # Connect button clicks to functions
        submit_btn.click(
            fn=process_input,
            inputs=[user_input],
            outputs=[response_output, history_output]
        ).then(
            fn=lambda: "",
            inputs=[],
            outputs=[user_input]
        )
        
        clear_btn.click(
            fn=clear_chat,
            inputs=[],
            outputs=[response_output, history_output]
        )
        
        # Allow Enter key to submit
        user_input.submit(
            fn=process_input,
            inputs=[user_input],
            outputs=[response_output, history_output]
        ).then(
            fn=lambda: "",
            inputs=[],
            outputs=[user_input]
        )
    
    return interface


def main():
    """Main entry point."""
    print("Starting Historical Figures Chatbot...")
    print("Launching Gradio interface...")
    
    interface = create_gradio_interface()
    interface.launch(
        server_port=7860,
        share=False,
        debug=True
    )


if __name__ == "__main__":
    main()
