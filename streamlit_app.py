# app.py
import streamlit as st
import os
import tempfile
from rag_system import RAGSystem
import time

# Page config
st.set_page_config(
    page_title="PDF RAG System",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def load_rag_system():
    """Load the RAG system (cached to avoid reloading)"""
    return RAGSystem()

def main():
    st.title("📚 PDF RAG System")
    st.markdown("Upload PDFs and ask questions about their content using AI!")
    
    # Initialize RAG system
    rag = load_rag_system()
    
    # Sidebar for file upload and stats
    with st.sidebar:
        st.header("📄 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if st.button(f"Process {uploaded_file.name}"):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_file_path = tmp_file.name
                        
                        try:
                            chunks_added = rag.process_pdf(tmp_file_path)
                            st.success(f"✅ Processed {uploaded_file.name}: {chunks_added} chunks added")
                        except Exception as e:
                            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                        finally:
                            # Clean up temp file
                            os.unlink(tmp_file_path)
        
        # System stats
        st.header("📊 System Stats")
        stats = rag.get_stats()
        st.metric("Total Chunks", stats['total_chunks'])
        
        if stats['sources']:
            st.subheader("📑 Loaded Documents")
            for source in stats['sources']:
                st.write(f"• {source}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Questions")
        
        # Chat interface
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(message["sources"][:3], 1):
                            st.write(f"**{i}. {source['source']}** (Page {source['page_num']})")
                            st.write(f"Similarity: {source['similarity_score']:.3f}")
                            st.write(f"*{source['text'][:200]}...*")
                            st.divider()
        
        # Chat input
if prompt := st.chat_input("Ask a question about your PDFs..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = rag.query(prompt)
                    st.write(result['answer'])
                    
                    # Show sources
                    if result['sources']:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(result['sources'][:3], 1):
                                st.write(f"**{i}. {source['source']}** (Page {source['page_num']})")
                                st.write(f"Similarity: {source['similarity_score']:.3f}")
                                st.write(f"*{source['text'][:200]}...*")
                                st.divider()
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result['answer'],
                "sources": result['sources']
            })
    
    with col2:
        st.header("🔧 System Info")
        
        # Model info
        st.subheader("🤖 Models Used")
        st.write("**Embeddings:** all-MiniLM-L6-v2")
        st.write("**LLM:** Llama 3.2 (via Ollama)")
        st.write("**Vector DB:** FAISS")
        
        # Instructions
        st.subheader("📋 Instructions")
        st.markdown("""
        1. **Upload PDFs** using the sidebar
        2. **Process** each PDF by clicking the button
        3. **Ask questions** in the chat interface
        4. View **sources** for each answer
        
        **Prerequisites:**
        - Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
        - Pull model: `ollama pull llama3.2`
        """)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()
