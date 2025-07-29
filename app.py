# app.py
"""
Modern PDF Chatbot with Contemporary UI Design
Features: Glass morphism, animated components, dark theme, and premium interactions
"""
import streamlit as st
import requests
from datetime import datetime
import os
import sys
import time
import json

# Add the directory containing chatbot.py to the path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the core logic
try:
    from chatbot import PDFChatbot, check_ollama_status
except ImportError as e:
    st.error(f"Failed to import PDFChatbot from chatbot.py: {e}")
    st.stop()

def inject_modern_css():
    """Inject modern CSS with glassmorphism, animations, and contemporary design"""
    st.markdown("""
    <style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }

    /* Hide default elements */
    header[data-testid="stHeader"] { display: none !important; }
    .stDeployButton { display: none !important; }
    #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden !important; }

    /* Custom header */
    .modern-header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        animation: slideDown 0.8s ease-out;
    }

    .modern-header h1 {
        background: linear-gradient(135deg, #fff 0%, #f0f0f0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }

    .modern-header p {
        color: rgba(255, 255, 255, 0.8);
        font-size: 1.2rem;
        font-weight: 300;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(20, 20, 40, 0.95) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Glass cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
    }

    .glass-card:hover {
        background: rgba(255, 255, 255, 0.15);
        transform: translateY(-2px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
    }

    /* Status indicators */
    .status-good {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 500;
        display: inline-block;
        animation: pulse 2s infinite;
    }

    .status-error {
        background: linear-gradient(135deg, #f44336, #d32f2f);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 500;
        display: inline-block;
    }

    /* Chat interface */
    .chat-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        min-height: 400px;
    }

    /* Chat messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(10px);
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        margin: 0.5rem 0 !important;
        animation: messageSlide 0.5s ease-out;
    }

    /* User message styling */
    .stChatMessage[data-testid*="user"] {
        background: linear-gradient(135deg, rgba(103, 126, 234, 0.3), rgba(118, 75, 162, 0.3)) !important;
        border: 1px solid rgba(103, 126, 234, 0.5) !important;
    }

    /* Assistant message styling */
    .stChatMessage[data-testid*="assistant"] {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%) !important;
    }

    /* File uploader */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 2px dashed rgba(255, 255, 255, 0.3);
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .stFileUploader:hover {
        border-color: rgba(102, 126, 234, 0.8);
        background: rgba(102, 126, 234, 0.1);
    }

    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
        border-radius: 10px !important;
    }

    /* Metrics */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: scale(1.05);
        background: rgba(255, 255, 255, 0.15);
    }

    /* Animations */
    @keyframes slideDown {
        from { transform: translateY(-50px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    @keyframes fadeInUp {
        from { transform: translateY(30px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    @keyframes messageSlide {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }

    /* Glow effects */
    .glow {
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
        animation: glow 2s ease-in-out infinite alternate;
    }

    @keyframes glow {
        from { box-shadow: 0 0 20px rgba(102, 126, 234, 0.3); }
        to { box-shadow: 0 0 30px rgba(102, 126, 234, 0.6); }
    }

    /* Text colors */
    .stMarkdown, .stText, p, div, span {
        color: rgba(255, 255, 255, 0.9) !important;
    }

    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        color: white !important;
    }

    /* Input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        color: white !important;
    }

    /* Chat input */
    .stChatInput > div {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 25px !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }

    /* Success/Error messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }

    /* Sidebar text */
    .css-1d391kg .stMarkdown,
    .css-1d391kg .stText,
    .css-1d391kg p,
    .css-1d391kg div,
    .css-1d391kg span {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    </style>
    """, unsafe_allow_html=True)

def display_modern_header():
    """Display modern animated header"""
    st.markdown("""
    <div class="modern-header glow">
        <h1>🤖 AI PDF Assistant</h1>
        <p>Transform your documents into intelligent conversations</p>
    </div>
    """, unsafe_allow_html=True)

def display_pdf_metadata(metadata):
    """Display PDF metadata with modern cards"""
    if metadata:
        st.markdown("---")
        st.markdown("### 📊 Document Analytics")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0; color: #667eea;">📄</h3>
                <h2 style="margin:0.5rem 0;">{metadata.get('total_pages', 'N/A')}</h2>
                <p style="margin:0; opacity:0.8;">Pages</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0; color: #764ba2;">🧩</h3>
                <h2 style="margin:0.5rem 0;">{metadata.get('chunks_created', 'N/A')}</h2>
                <p style="margin:0; opacity:0.8;">Chunks</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            chars = metadata.get('total_characters', 0)
            formatted_chars = f"{chars:,}" if chars else 'N/A'
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0; color: #4CAF50;">📝</h3>
                <h2 style="margin:0.5rem 0;">{formatted_chars}</h2>
                <p style="margin:0; opacity:0.8;">Characters</p>
            </div>
            """, unsafe_allow_html=True)

        st.caption(f"⏰ Processed: {metadata.get('processed_at', 'Unknown')}")

def display_sources(sources, expander_title="🔍 Knowledge Sources"):
    """Display sources with modern styling"""
    if sources:
        with st.expander(expander_title):
            for i, doc in enumerate(sources[:3], 1):
                st.markdown(f"**📑 Source {i}:**")
                content_preview = doc.page_content[:400]
                if len(doc.page_content) > 400:
                    content_preview += "..."

                st.markdown(f"""
                <div style="
                    background: rgba(255, 255, 255, 0.05);
                    border-left: 3px solid #667eea;
                    padding: 1rem;
                    border-radius: 8px;
                    margin: 0.5rem 0;
                    font-family: 'SF Mono', monospace;
                ">
                    {content_preview}
                </div>
                """, unsafe_allow_html=True)

def display_ollama_status():
    """Display Ollama status with modern indicators"""
    with st.status("🔍 System Check", expanded=True) as status:
        ollama_running, available_models = check_ollama_status()

        if ollama_running:
            st.markdown('<div class="status-good">✨ Ollama Connected</div>', unsafe_allow_html=True)

            if available_models:
                model_names = [model.get("name", "") for model in available_models if model.get("name")]
                st.success(f"🧠 {len(model_names)} AI model(s) ready")

                # Model selection with modern styling
                default_model = "llama3.2:3b"
                default_index = model_names.index(default_model) if default_model in model_names else 0

                selected_model = st.selectbox(
                    "🎯 Choose AI Model:",
                    options=model_names,
                    index=default_index,
                    key="model_selector"
                )
                return selected_model, True
            else:
                st.warning("⚠️ No models detected")
                st.info("💡 Try: `ollama pull llama3.2:3b`")
                selected_model = st.text_input("Manual model:", value="llama3.2:3b", key="manual_model")
                return selected_model, False
        else:
            st.markdown('<div class="status-error">❌ Ollama Offline</div>', unsafe_allow_html=True)
            st.error("🔧 Start with: `ollama serve`")
            selected_model = st.text_input("Fallback model:", value="llama3.2:3b", key="fallback_model")
            return selected_model, False

def main():
    # Configuration
    st.set_page_config(
        page_title="AI PDF Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inject modern CSS
    inject_modern_css()

    # Display modern header
    display_modern_header()

    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = PDFChatbot()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = {}

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Control Panel")

        # Ollama status
        selected_model, models_available = display_ollama_status()

        st.markdown("---")

        # File upload section
        st.markdown("### 📤 Document Upload")
        uploaded_file = st.file_uploader(
            "Drop your PDF here",
            type="pdf",
            accept_multiple_files=False,
            key="pdf_uploader"
        )

        if uploaded_file is not None:
            # File info with modern styling
            file_size_kb = uploaded_file.size / 1024
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="margin-top:0;">📋 File Info</h4>
                <p><strong>Name:</strong> {uploaded_file.name}</p>
                <p><strong>Size:</strong> {file_size_kb:.1f} KB</p>
            </div>
            """, unsafe_allow_html=True)

            # Process button
            process_button = st.button(
                "🚀 Process Document",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.get('processing_in_progress', False)
            )

            if process_button:
                st.session_state.processing_in_progress = True
                st.session_state.processing_status = {'extract': False, 'embed': False, 'chain': False}

                try:
                    with st.status("🔄 Processing Magic...", expanded=True) as status:
                        # Step 1: Extract Text
                        status.update(label="📖 Reading document...", state="running")
                        try:
                            text = st.session_state.chatbot.extract_text_from_pdf(uploaded_file)
                            if text:
                                st.session_state.processing_status['extract'] = True
                                status.update(label="✅ Text extracted", state="complete")
                            else:
                                raise Exception("No text found")
                        except Exception as e:
                            st.session_state.processing_status['extract'] = False
                            status.update(label="❌ Extraction failed", state="error")
                            st.error(f"Text extraction error: {e}")
                            st.session_state.processing_in_progress = False
                            st.stop()

                        # Step 2: Create embeddings
                        if st.session_state.processing_status['extract']:
                            status.update(label="🧠 Creating neural embeddings...", state="running")
                            try:
                                if st.session_state.chatbot.create_vector_store(text):
                                    st.session_state.processing_status['embed'] = True
                                    status.update(label="✅ Embeddings ready", state="complete")
                                else:
                                    raise Exception("Vector store failed")
                            except Exception as e:
                                st.session_state.processing_status['embed'] = False
                                status.update(label="❌ Embedding failed", state="error")
                                st.error(f"Embedding error: {e}")
                                st.session_state.processing_in_progress = False
                                st.stop()

                        # Step 3: Setup AI chain
                        if st.session_state.processing_status['embed']:
                            status.update(label="🔗 Connecting AI brain...", state="running")
                            try:
                                if st.session_state.chatbot.setup_chain(model_name=selected_model):
                                    st.session_state.processing_status['chain'] = True
                                    st.session_state.pdf_processed = True
                                    status.update(label="🎉 AI Assistant Ready!", state="complete", expanded=False)
                                    st.success("🎊 Document processed! Start chatting below.")
                                    st.rerun()
                                else:
                                    raise Exception("Chain setup failed")
                            except Exception as e:
                                st.session_state.processing_status['chain'] = False
                                status.update(label="❌ AI setup failed", state="error")
                                st.error(f"AI setup error: {e}")

                                # Intelligent error suggestions
                                error_str = str(e).lower()
                                if "connection" in error_str:
                                    st.info("💡 Solution: `ollama serve`")
                                elif "model" in error_str:
                                    st.info(f"💡 Solution: `ollama pull {selected_model}`")

                                st.session_state.processing_in_progress = False
                                st.stop()

                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                finally:
                    st.session_state.processing_in_progress = False

        # Display PDF metadata
        display_pdf_metadata(st.session_state.chatbot.pdf_metadata)

        # Clear chat button
        st.markdown("---")
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Main chat interface
    st.markdown("---")

    # Chat container
    if st.session_state.pdf_processed:
        st.markdown("### 💬 AI Conversation")

    # Display messages
    if st.session_state.messages:
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "sources" in message:
                    display_sources(message["sources"])

    # Chat input
    if st.session_state.pdf_processed:
        prompt = st.chat_input(
            "✨ Ask anything about your document...",
            key="chat_input"
        )

        if prompt:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)

            # Get AI response
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        answer, sources = st.session_state.chatbot.chat(prompt)
                        st.markdown(answer)
                        display_sources(sources)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                    except Exception as e:
                        error_msg = f"🚨 Error: {e}"
                        st.markdown(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "sources": []
                        })
    else:
        # Welcome screen
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 3rem;">
            <h2 style="margin-bottom: 1rem;">🚀 Ready to Get Started?</h2>
            <p style="font-size: 1.1rem; margin-bottom: 2rem; opacity: 0.8;">
                Upload a PDF document and unlock the power of AI-driven conversations
            </p>
            <div style="font-size: 4rem; margin: 2rem 0;">📚➡️🤖</div>
        </div>
        """, unsafe_allow_html=True)

        # Feature showcase
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">⚡</div>
                <h3>Lightning Fast</h3>
                <p>Instant document processing with advanced AI embeddings</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">🎯</div>
                <h3>Precise Answers</h3>
                <p>Get accurate responses with source references</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">🧠</div>
                <h3>Smart Memory</h3>
                <p>Contextual conversations that remember previous exchanges</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
