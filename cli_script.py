#!/usr/bin/env python3
"""
Simplified CLI runner for the PDF RAG System
"""
import sys
import os
import subprocess
from pathlib import Path

def print_banner():
    """Print the application banner"""
    print("""
    ╔══════════════════════════════════════╗
    ║         PDF RAG System v1.0          ║
    ║  AI-Powered Document Q&A System      ║
    ╚══════════════════════════════════════╝
    """)

def check_dependencies():
    """Check if required dependencies are available"""
    issues = []
    
    # Check Python packages
    try:
        import sentence_transformers
        import faiss
        import streamlit
        import fitz  # PyMuPDF
    except ImportError as e:
        issues.append(f"Missing Python package: {e.name}")
    
    # Check Ollama
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, check=True)
        if "llama3.2" not in result.stdout.decode():
            issues.append("Llama 3.2 model not found. Run: ollama pull llama3.2")
    except (subprocess.CalledProcessError, FileNotFoundError):
        issues.append("Ollama not found or not running. Install from https://ollama.ai/")
    
    return issues

def main():
    print_banner()
    
    # Check dependencies
    issues = check_dependencies()
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"   • {issue}")
        print("\n💡 Run 'python setup.py' to fix these issues")
        sys.exit(1)
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run.py web                    # Start web interface")
        print("  python run.py cli                    # Start CLI interface")
        print("  python run.py upload <pdf_file>      # Upload a PDF")
        print("  python run.py query '<question>'     # Ask a question")
        print("  python run.py stats                  # Show system stats")
        print("  python run.py test                   # Run tests")
        return
    
    command = sys.argv[1].lower()
    
    if command == "web":
        print("🌐 Starting web interface...")
        os.system("streamlit run app.py")
    
    elif command == "cli":
        print("💻 Starting CLI interface...")
        os.system("python rag_system.py --interactive")
    
    elif command == "upload":
        if len(sys.argv) < 3:
            print("❌ Please provide a PDF file path")
            return
        pdf_file = sys.argv[2]
        print(f"📄 Uploading {pdf_file}...")
        os.system(f"python rag_system.py --upload '{pdf_file}'")
    
    elif command == "query":
        if len(sys.argv) < 3:
            print("❌ Please provide a question")
            return
        question = " ".join(sys.argv[2:])
        print(f"❓ Asking: {question}")
        os.system(f"python rag_system.py --query '{question}'")
    
    elif command == "stats":
        print("📊 System statistics:")
        os.system("python rag_system.py --stats")
    
    elif command == "test":
        print("🧪 Running tests...")
        os.system("python -m pytest tests/ -v")
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Use 'python run.py' without arguments to see available commands")

if __name__ == "__main__":
    main()
