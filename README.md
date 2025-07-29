# AI PDF Assistant

A modern PDF chatbot application powered by Streamlit, LangChain, and Ollama. This project allows users to upload PDF documents and interact with their content through an intelligent conversational interface. The application features a sleek, contemporary UI with glassmorphism design, animations, and a dark theme.

## Features

- **PDF Processing**: Extract text from PDF files using `pdfplumber`.
- **Intelligent Chat**: Leverage LangChain and Ollama for contextual conversations with document content.
- **Modern UI**: Glassmorphism design, animated components, and a dark theme for a premium user experience.
- **Vector Search**: Uses FAISS and HuggingFace embeddings for efficient document retrieval.
- **Robust Error Handling**: Comprehensive dependency management and fallback mechanisms.
- **Ollama Integration**: Supports local LLMs with easy model selection.

## Prerequisites

- **Python**: Version 3.8 or higher.
- **Ollama**: Must be installed and running locally (`ollama serve`).
- **PDF Documents**: Any PDF file to interact with the chatbot.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/jeewnish/pdf-chatbot.git
   cd pdf-chatbot
   ```

2. **Install Dependencies**:
   Run the provided dependency fix script to ensure all required packages are installed correctly:
   ```bash
   python launcher.py
   ```

   This script:
   - Checks Python version compatibility.
   - Removes conflicting packages.
   - Installs all dependencies in the correct order.
   - Creates a `requirements_fixed.txt` file.
   - Verifies installations and imports.

3. **Start Ollama Server**:
   Ensure the Ollama server is running:
   ```bash
   ollama serve
   ```

4. **Download a Model**:
   Pull a compatible model (e.g., Llama 3.2):
   ```bash
   ollama pull llama3.2:3b
   ```

## Usage

1. **Run the Application**:
   Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. **Interact with the App**:
   - Open the app in your browser (typically `http://localhost:8501`).
   - Upload a PDF file via the sidebar.
   - Click "Process Document" to extract text and create embeddings.
   - Once processed, use the chat interface to ask questions about the document.
   - View document analytics and source references for answers.

## Project Structure

- `launcher.py`: Dependency management script to fix and install all required packages.
- `chatbot.py`: Core logic module for PDF processing, vector store creation, and chat functionality.
- `app.py`: Streamlit UI with modern design and user interaction handling.
- `requirements_fixed.txt`: Generated file listing all required dependencies with fixed versions.

## Dependencies

Key dependencies include:
- `streamlit==1.40.0`: For the web interface.
- `pdfplumber==0.11.4`: For PDF text extraction.
- `langchain==0.3.14` & related packages: For conversational AI.
- `faiss-cpu==1.11.0.post1`: For vector storage and search.
- `sentence-transformers==3.0.1`: For text embeddings.
- `torch==2.5.1+cpu`: CPU-only PyTorch to avoid GPU conflicts.
- `requests==2.32.3`, `numpy==1.26.4`, `pandas==2.2.2`: Utilities.

See `requirements_fixed.txt` for the complete list.

## Troubleshooting

- **Ollama Not Running**:
  - Ensure the Ollama server is active: `ollama serve`.
  - Pull a model if none are available: `ollama pull llama3.2:3b`.
- **Dependency Issues**:
  - Rerun `launcher.py` to fix package conflicts.
  - Check Python version (3.8+ required).
- **PDF Processing Errors**:
  - Ensure the PDF is not corrupted or password-protected.
  - Verify sufficient disk space and memory.
- **Import Errors**:
  - Run `launcher.py` to verify and reinstall dependencies.

## Contributing

Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/), [LangChain](https://langchain.com/), and [Ollama](https://ollama.ai/).
- Inspired by modern UI design trends and AI-driven document processing.
