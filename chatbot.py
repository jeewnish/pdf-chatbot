# chatbot.py
"""
PDF Chatbot Core Logic Module
This module contains the PDFChatbot class with methods for processing PDFs,
creating vector stores, setting up the chat chain, and handling chat interactions.
All Streamlit UI dependencies have been removed from core logic methods.
"""
import os
import warnings
import sys
import logging
from datetime import datetime

# Complete warning suppression setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PYTHONWARNINGS"] = "ignore"
# Suppress all warnings
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
# Suppress torch-specific warnings
try:
    import torch
    torch.set_num_threads(1)
except:
    pass

# Core imports (no streamlit here)
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import time
import requests

# Import the correct embeddings class with multiple fallbacks
embeddings_class = None
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings_class = HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings_class = HuggingFaceEmbeddings
    except ImportError:
        # Raise an exception that the calling UI can catch and handle appropriately
        raise ImportError("Could not import HuggingFaceEmbeddings. Please install: pip install langchain-huggingface")

# Import OllamaLLM with multiple fallbacks
llm_class = None
try:
    from langchain_ollama import OllamaLLM
    llm_class = OllamaLLM
except ImportError:
    try:
        from langchain_community.llms import Ollama
        llm_class = Ollama
    except ImportError:
        # Raise an exception that the calling UI can catch and handle appropriately
        raise ImportError("Could not import OllamaLLM. Please install: pip install langchain-ollama")

class PDFChatbot:
    def __init__(self):
        self.vectorstore = None
        self.chain = None
        self.pdf_metadata = {}

    def extract_text_from_pdf(self, pdf_file):
        """
        Extract text from PDF with robust error handling.
        Returns:
            str: Extracted text.
        Raises:
            Exception: If PDF reading or text extraction fails.
        """
        text = ""
        try:
            # Reset metadata for new file
            self.pdf_metadata = {
                "total_pages": 0,
                "file_name": getattr(pdf_file, 'name', 'Unknown'),
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_characters": 0,
                "chunks_created": 0
            }

            with pdfplumber.open(pdf_file) as pdf:
                self.pdf_metadata["total_pages"] = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n[Page {page_num}]\n{page_text}\n"
                    except Exception as e:
                        # Log or print warning if needed, but don't use st.warning
                        print(f"Warning: Could not extract text from page {page_num}: {e}")
                        continue # Continue processing other pages
        except Exception as e:
            raise Exception(f"Error reading PDF file: {e}")

        if not text.strip():
            raise Exception("No text could be extracted from the PDF")

        self.pdf_metadata["total_characters"] = len(text)
        return text

    def create_vector_store(self, text):
        """
        Create vector store with multiple embedding model fallbacks.
        Args:
            text (str): The text to embed and store.
        Returns:
            bool: True if successful.
        Raises:
            Exception: If text processing, embedding, or vector store creation fails.
        """
        if not text.strip():
            raise Exception("No text found in PDF")

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        if not chunks:
            raise Exception("No text chunks created")

        # Try different embedding models in order of preference
        embedding_models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/distilbert-base-nli-mean-tokens"
        ]
        embeddings = None
        last_exception = None
        for model_name in embedding_models:
            try:
                # Create embeddings with minimal parameters to avoid conflicts
                embeddings = embeddings_class(
                    model_name=model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True}
                )
                # Test the embeddings with a simple query
                test_embedding = embeddings.embed_query("test")
                if test_embedding and len(test_embedding) > 0:
                    # Successfully loaded
                    break
                else:
                    raise Exception("Embedding test returned empty or invalid result")
            except Exception as e:
                last_exception = e
                embeddings = None
                # Continue trying the next model
                continue

        if embeddings is None:
            raise Exception(f"All embedding models failed. Last error: {last_exception}. Please check your internet connection.") from last_exception

        # Create vector store
        try:
            self.vectorstore = FAISS.from_texts(chunks, embeddings)
            self.pdf_metadata["chunks_created"] = len(chunks)
            return True
        except Exception as e:
            raise Exception(f"Error creating vector store: {e}")

    def setup_chain(self, model_name="llama3.2:3b", temperature=0.7):
        """
        Setup conversational chain with robust error handling.
        Args:
            model_name (str): The name of the Ollama model to use.
            temperature (float): The temperature for the LLM.
        Returns:
            bool: True if successful.
        Raises:
            Exception: If Ollama connection fails, LLM initialization fails, or chain creation fails.
        """
        if not self.vectorstore:
            raise Exception("Vector store not created. Please process a PDF first.")

        # Test Ollama connection first
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                raise Exception("Ollama server is not responding (status code != 200). Please start it with: ollama serve")
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Ollama server. Please start it with: ollama serve")
        except Exception as e:
            raise Exception(f"Error checking Ollama server: {e}")

        # Initialize LLM with multiple strategies
        llm = None
        initialization_strategies = [
            # Strategy 1: Full parameters
            {
                "model": model_name,
                "temperature": temperature,
                "timeout": 60,
                "base_url": "http://localhost:11434"
            },
            # Strategy 2: Minimal parameters
            {
                "model": model_name,
                "temperature": temperature
            },
            # Strategy 3: Basic initialization
            {
                "model": model_name
            }
        ]
        last_exception = None
        for i, kwargs in enumerate(initialization_strategies, 1):
            try:
                llm = llm_class(**kwargs)
                # Test the LLM with a simple query
                test_response = llm.invoke("Hi")
                if test_response and len(str(test_response).strip()) > 0:
                    # Successfully working
                    break
                else:
                    raise Exception("Empty response from LLM test")
            except Exception as e:
                last_exception = e
                llm = None
                # Continue trying the next strategy
                continue

        if llm is None:
            raise Exception(f"All LLM initialization strategies failed. Last error: {last_exception}. Please ensure Ollama is running and the model '{model_name}' is available.") from last_exception

        # Setup conversation memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        # Create conversational retrieval chain
        try:
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                ),
                memory=memory,
                return_source_documents=True,
                verbose=False
            )
            return True
        except Exception as e:
            raise Exception(f"Chain setup failed: {e}")

    def chat(self, question):
        """
        Chat with the PDF content.
        Args:
            question (str): The user's question.
        Returns:
            tuple: A tuple containing (answer (str), sources (list)).
        Raises:
            Exception: If the chain is not set up or an error occurs during chat.
        """
        if not self.chain:
            raise Exception("Chat chain is not set up. Please upload and process a PDF first.")

        try:
            # Invoke the chain with the question
            response = self.chain({"question": question})
            # Extract answer and sources
            answer = response.get("answer", "No answer generated")
            sources = response.get("source_documents", [])
            return answer, sources
        except Exception as e:
            raise Exception(f"Chat error: {e}")

# The check_ollama_status function can remain here if needed by other modules,
# or be moved to app.py. Let's keep it here for now as it's utility logic.
def check_ollama_status():
    """Check if Ollama is running and has models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            # Handle potential structures like {"models": [...]} or directly [...]
            if isinstance(models_data, dict):
                models_list = models_data.get("models", [])
            elif isinstance(models_data, list):
                models_list = models_data
            else:
                models_list = []
            return True, models_list
        else:
            return False, []
    except requests.exceptions.RequestException:
        return False, []
    except Exception:
        # Fallback if JSON parsing fails or other unexpected errors
        return False, []

# The main() function is removed as it contained Streamlit UI code.
# The Streamlit app logic will reside in app.py

if __name__ == "__main__":
    # If run directly, perhaps print a message or provide CLI interface in the future
    print("This module is intended to be imported by a Streamlit app (e.g., app.py).")
    print("Use 'streamlit run app.py' to start the application.")
