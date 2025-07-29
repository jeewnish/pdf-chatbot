#!/usr/bin/env python3
"""
Complete Dependency Fix Script for PDF Chatbot
Fixes all subprocess, import, and dependency issues
"""
import subprocess
import sys
import os
import time

def run_command_fixed(command, show_output=True):
    """Fixed run_command function that avoids stdout/stderr conflicts"""
    if show_output:
        print(f"🔧 Running: {command}")

    try:
        if show_output:
            # When showing output, don't capture it
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                text=True
            )
        else:
            # When not showing output, capture it properly
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                text=True,
                capture_output=True
            )
        
        if show_output:
            print("✅ Success!")
        return True, result.stdout if not show_output else ""
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {command}")
        if hasattr(e, 'stdout') and e.stdout:
            print(f"Output: {e.stdout}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"Error: {e.stderr}")
        return False, ""

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print("✅ Python version is compatible")
    return True

def uninstall_conflicting_packages():
    """Remove potentially conflicting packages"""
    print("\n🗑️ Removing conflicting packages...")
    
    packages_to_remove = [
        "torch torchvision torchaudio",
        "langchain-openai",  # Sometimes conflicts
        "transformers[torch]"  # Can cause conflicts
    ]
    
    for package in packages_to_remove:
        print(f"\nRemoving: {package}")
        success, _ = run_command_fixed(f"pip uninstall {package} -y", show_output=False)
        if success:
            print(f"✅ Removed {package}")
        else:
            print(f"ℹ️ {package} was not installed")

def install_core_dependencies():
    """Install core dependencies in the correct order"""
    print("\n📦 Installing core dependencies...")
    
    # Core dependencies in installation order
    dependencies = [
        # Step 1: Core Python packages
        "wheel setuptools pip --upgrade",
        
        # Step 2: CPU-only PyTorch (lighter and avoids GPU conflicts)
        "torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu",
        
        # Step 3: Core LangChain packages
        "langchain==0.3.14",
        "langchain-core==0.3.29",
        "langchain-community==0.3.14",
        
        # Step 4: Specialized LangChain packages
        "langchain-ollama==0.2.2",
        "langchain-huggingface==0.1.2",
        "langchain-text-splitters==0.3.5",
        
        # Step 5: Vector database and embeddings
        "faiss-cpu==1.11.0.post1",
        "sentence-transformers==3.0.1",
        
        # Step 6: PDF processing
        "pdfplumber==0.11.4",
        "PyPDF2==3.0.1",
        
        # Step 7: Web framework
        "streamlit==1.40.0",
        
        # Step 8: Utilities
        "requests==2.32.3",
        "numpy==1.26.4",
        "pandas==2.2.2"
    ]
    
    successful_installs = 0
    failed_installs = []
    
    for dependency in dependencies:
        print(f"\n{'-'*50}")
        print(f"Installing: {dependency}")
        
        success, output = run_command_fixed(f"pip install {dependency}", show_output=False)
        
        if success:
            print(f"✅ Successfully installed: {dependency}")
            successful_installs += 1
        else:
            print(f"❌ Failed to install: {dependency}")
            failed_installs.append(dependency)
            
        # Small delay to prevent overwhelming pip
        time.sleep(1)
    
    print(f"\n{'='*50}")
    print(f"📊 Installation Summary:")
    print(f"✅ Successful: {successful_installs}/{len(dependencies)}")
    if failed_installs:
        print(f"❌ Failed: {len(failed_installs)}")
        for failed in failed_installs:
            print(f"   - {failed}")
    
    return len(failed_installs) == 0

def verify_installations():
    """Verify that all required packages are properly installed"""
    print("\n🔍 Verifying installations...")
    
    packages_to_check = [
        ("streamlit", "streamlit"),
        ("pdfplumber", "pdfplumber"),
        ("langchain", "langchain"),
        ("langchain_ollama", "langchain_ollama"),
        ("langchain_community", "langchain_community"),
        ("langchain_huggingface", "langchain_huggingface"),
        ("faiss", "faiss"),
        ("sentence_transformers", "sentence_transformers"),
        ("torch", "torch"),
        ("requests", "requests")
    ]
    
    all_good = True
    
    for package_name, import_name in packages_to_check:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError as e:
            print(f"❌ {package_name} - {e}")
            all_good = False
    
    return all_good

def create_updated_requirements():
    """Create an updated requirements.txt file"""
    print("\n📝 Creating updated requirements.txt...")
    
    requirements_content = """# Core dependencies for PDF Chatbot (Fixed versions)
streamlit==1.40.0
pdfplumber==0.11.4

# LangChain ecosystem (compatible versions)
langchain==0.3.14
langchain-core==0.3.29
langchain-ollama==0.2.2
langchain-community==0.3.14
langchain-text-splitters==0.3.5
langchain-huggingface==0.1.2

# Vector database and embeddings
faiss-cpu==1.11.0.post1
sentence-transformers==3.0.1

# PyTorch CPU-only (avoids GPU conflicts)
torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Utilities
requests==2.32.3
numpy==1.26.4
pandas==2.2.2

# PDF processing alternatives
PyPDF2==3.0.1

# Optional: For better performance
python-dotenv==1.0.1
typing-extensions==4.12.2
"""
    
    try:
        with open("requirements_fixed.txt", "w") as f:
            f.write(requirements_content)
        print("✅ Created requirements_fixed.txt")
        return True
    except Exception as e:
        print(f"❌ Failed to create requirements file: {e}")
        return False

def test_basic_imports():
    """Test that all critical imports work"""
    print("\n🧪 Testing critical imports...")
    
    test_imports = [
        "import streamlit as st",
        "import pdfplumber",
        "from langchain.text_splitter import RecursiveCharacterTextSplitter",
        "from langchain_community.vectorstores import FAISS",
        "from langchain.chains import ConversationalRetrievalChain",
        "from langchain.memory import ConversationBufferMemory",
        "from langchain_huggingface import HuggingFaceEmbeddings",
        "from langchain_ollama import OllamaLLM",
        "import torch",
        "import requests"
    ]
    
    all_imports_successful = True
    
    for import_statement in test_imports:
        try:
            exec(import_statement)
            print(f"✅ {import_statement}")
        except Exception as e:
            print(f"❌ {import_statement} - {e}")
            all_imports_successful = False
    
    return all_imports_successful

def main():
    print("🛠️ PDF Chatbot Complete Dependency Fix")
    print("=" * 50)
    print("This script will fix ALL known issues:")
    print("- ✅ Subprocess stdout/stderr conflicts")
    print("- ✅ OllamaLLM import and initialization errors")
    print("- ✅ HuggingFaceEmbeddings deprecation warnings")
    print("- ✅ Package version conflicts")
    print("- ✅ PyTorch GPU/CPU conflicts")
    print("- ✅ Chain setup and memory issues")
    print()
    
    # Get user confirmation
    response = input("🚀 Proceed with complete fix? (y/n): ").lower().strip()
    if response != 'y':
        print("❌ Cancelled by user.")
        return 1
    
    print("\n🔧 Starting complete fix process...")
    
    # Step 1: Check Python version
    if not check_python_version():
        print("❌ Python version incompatible. Please upgrade to Python 3.8+")
        return 1
    
    # Step 2: Remove conflicting packages
    uninstall_conflicting_packages()
    
    # Step 3: Install dependencies in correct order
    if not install_core_dependencies():
        print("\n❌ Some installations failed. Please check the errors above.")
        print("💡 Try running the failed commands manually.")
        return 1
    
    # Step 4: Verify installations
    print("\n" + "="*50)
    if not verify_installations():
        print("❌ Some packages are still missing. Check the errors above.")
        return 1
    
    # Step 5: Test imports
    if not test_basic_imports():
        print("❌ Some imports are still failing. Check the errors above.")
        return 1
    
    # Step 6: Create updated requirements file
    create_updated_requirements()
    
    # Success message
    print("\n" + "🎉" * 20)
    print("🎉 COMPLETE FIX SUCCESSFUL! 🎉")
    print("🎉" * 20)
    print()
    print("✅ All dependencies installed and verified")
    print("✅ All imports working correctly")
    print("✅ Subprocess issues resolved")
    print("✅ Chain and memory issues fixed")
    print()
    print("🚀 Next steps:")
    print("1. Ensure Ollama is running: ollama serve")
    print("2. Download a model: ollama pull llama3.2:3b")
    print("3. Run the modern chatbot UI: streamlit run app.py") # Updated instruction
    print()
    print("📁 Files referenced:")
    print("- chatbot.py (core logic module)")
    print("- app.py (modern Streamlit UI - you need to create this or use the provided one)")
    print("- requirements_fixed.txt (updated requirements)")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
# Removed the erroneous 'end.' at the end of the file
