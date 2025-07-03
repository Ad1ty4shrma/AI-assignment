#!/usr/bin/env python3
"""
Setup script for PDF RAG System
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_ollama():
    """Install Ollama if not present"""
    try:
        subprocess.run(["ollama", "--version"], check=True, capture_output=True)
        print("✅ Ollama is already installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("📦 Installing Ollama...")
        if sys.platform == "darwin":  # macOS
            return run_command("curl -fsSL https://ollama.ai/install.sh | sh", "Installing Ollama")
        elif sys.platform == "linux":  # Linux
            return run_command("curl -fsSL https://ollama.ai/install.sh | sh", "Installing Ollama")
        else:
            print("⚠️  Please install Ollama manually from https://ollama.ai/")
            return False

def pull_llama_model():
    """Pull the Llama model"""
    return run_command("ollama pull llama3.2", "Pulling Llama 3.2 model")

def install_requirements():
    """Install Python requirements"""
    return run_command("pip install -r requirements.txt", "Installing Python packages")

def create_directories():
    """Create necessary directories"""
    directories = ['rag_storage', 'sample_pdfs', 'tests']
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"📁 Created directory: {dir_name}")

def download_sample_pdf():
    """Download a sample PDF for testing"""
    sample_pdf_path = Path("sample_pdfs/sample_document.pdf")
    if not sample_pdf_path.exists():
        print("📄 Creating sample PDF...")
        # Create a simple text file as a placeholder
        with open("sample_pdfs/sample_document.txt", "w") as f:
            f.write("""
Sample Document for RAG System Testing

Introduction
This is a sample document created for testing the PDF RAG system.
It contains multiple sections to demonstrate the chunking and retrieval capabilities.

Artificial Intelligence
Artificial Intelligence (AI) is the simulation of human intelligence in machines.
These systems can perform tasks that typically require human intelligence,
such as visual perception, speech recognition, and decision-making.

Machine Learning
Machine Learning is a subset of AI that enables computers to learn and improve
from experience without being explicitly programmed. It uses algorithms to
analyze data and make predictions or decisions.

Natural Language Processing
Natural Language Processing (NLP) is a branch of AI that helps computers
understand, interpret, and manipulate human language. It bridges the gap
between human communication and computer understanding.

Conclusion
This document provides a basic overview of AI concepts for testing purposes.
The RAG system should be able to answer questions about these topics.
            """)
        print("✅ Sample document created")

def main():
    """Main setup function"""
    print("🚀 Setting up PDF RAG System...")
    
    # Check Python version
    check_python_version()
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Install Ollama
    if not install_ollama():
        print("⚠️  Ollama installation failed - you may need to install it manually")
    
    # Pull Llama model
    if not pull_llama_model():
        print("⚠️  Failed to pull Llama model - you can do this later with 'ollama pull llama3.2'")
    
    # Download sample PDF
    download_sample_pdf()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Start Ollama: ollama serve")
    print("2. Run the web interface: streamlit run app.py")
    print("3. Or use CLI: python rag_system.py --help")
    print("4. Upload the sample document from sample_pdfs/")

if __name__ == "__main__":
    main()
