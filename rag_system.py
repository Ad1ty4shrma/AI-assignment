import os
import pickle
import json
from typing import List, Dict, Tuple
import numpy as np
import fitz  # PyMuPDF
import faiss
from sentence_transformers import SentenceTransformer
import requests
import re
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# PDF Processing
# ──────────────────────────────────────────────────────────────

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        doc = fitz.open(pdf_path)
        pages_text = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                pages_text.append({
                    'page_num': page_num + 1,
                    'text': text,
                    'source': os.path.basename(pdf_path)
                })

        doc.close()
        return pages_text

    def chunk_text(self, pages_text: List[Dict]) -> List[Dict]:
        chunks = []
        chunk_id = 0

        for page_data in pages_text:
            text = page_data['text']
            sentences = re.split(r'[.!?]+', text)
            current_chunk = ""

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if len(current_chunk) + len(sentence) > self.chunk_size:
                    if current_chunk:
                        chunks.append({
                            'id': chunk_id,
                            'text': current_chunk.strip(),
                            'page_num': page_data['page_num'],
                            'source': page_data['source']
                        })
                        chunk_id += 1
                        overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                        current_chunk = overlap_text + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk += " " + sentence if current_chunk else sentence

            if current_chunk.strip():
                chunks.append({
                    'id': chunk_id,
                    'text': current_chunk.strip(),
                    'page_num': page_data['page_num'],
                    'source': page_data['source']
                })
                chunk_id += 1

        return chunks

# ──────────────────────────────────────────────────────────────
# Embedding Generator
# ──────────────────────────────────────────────────────────────

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

    def generate_single_embedding(self, text: str) -> np.ndarray:
        return self.model.encode([text], convert_to_numpy=True)[0]

# ──────────────────────────────────────────────────────────────
# FAISS Vector Store
# ──────────────────────────────────────────────────────────────

class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = []

    def add_embeddings(self, embeddings: np.ndarray, chunks: List[Dict]):
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                result = self.chunks[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)

        return results

    def save(self, filepath: str):
        faiss.write_index(self.index, f"{filepath}.index")
        with open(f"{filepath}.chunks", 'wb') as f:
            pickle.dump(self.chunks, f)

    def load(self, filepath: str):
        self.index = faiss.read_index(f"{filepath}.index")
        with open(f"{filepath}.chunks", 'rb') as f:
            self.chunks = pickle.load(f)

# ──────────────────────────────────────────────────────────────
# LLM Client (Ollama)
# ──────────────────────────────────────────────────────────────

class LLMClient:
    def __init__(self, model_name: str = None, base_url: str = "http://localhost:11434"):
        self.model_name = "llama3"  # 🔥 Force it for now
        self.base_url = base_url


    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        context = "\n\n".join([
            f"[Source: {chunk['source']}, Page: {chunk['page_num']}]\n{chunk['text']}"
            for chunk in context_chunks
        ])

        prompt = f"""Based on the following context, answer the user's question. If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {query}

Answer:"""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )

            if response.status_code == 200:
                return response.json()['response']
            else:
                return f"Error: Unable to generate response. Status code: {response.status_code}"

        except requests.exceptions.RequestException as e:
            return f"Error: Could not connect to Ollama. Make sure it's running: {str(e)}"

# ──────────────────────────────────────────────────────────────
# RAG System
# ──────────────────────────────────────────────────────────────

class RAGSystem:
    def __init__(self, storage_dir: str = "rag_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        self.pdf_processor = PDFProcessor()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore(self.embedding_generator.dimension)
        self.llm_client = LLMClient()

        self.load_existing_index()

    def load_existing_index(self):
        index_path = self.storage_dir / "vector_store"
        if (Path(f"{index_path}.index")).exists():
            print("Loading existing vector store...")
            self.vector_store.load(str(index_path))
            print(f"Loaded {len(self.vector_store.chunks)} chunks")

    def save_index(self):
        index_path = self.storage_dir / "vector_store"
        self.vector_store.save(str(index_path))
        print(f"Saved vector store with {len(self.vector_store.chunks)} chunks")

    def process_pdf(self, pdf_path: str) -> int:
        print(f"Processing PDF: {pdf_path}")
        pages_text = self.pdf_processor.extract_text_from_pdf(pdf_path)
        if not pages_text:
            print("No text found in PDF")
            return 0

        chunks = self.pdf_processor.chunk_text(pages_text)
        print(f"Created {len(chunks)} chunks")
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_generator.generate_embeddings(chunk_texts)
        self.vector_store.add_embeddings(embeddings, chunks)
        self.save_index()
        return len(chunks)

    def query(self, question: str, top_k: int = 5) -> Dict:
        if len(self.vector_store.chunks) == 0:
            return {
                'answer': "No documents have been processed yet. Please upload some PDFs first.",
                'sources': []
            }

        query_embedding = self.embedding_generator.generate_single_embedding(question)
        relevant_chunks = self.vector_store.search(query_embedding, k=top_k)
        answer = self.llm_client.generate_answer(question, relevant_chunks)

        return {
            'answer': answer,
            'sources': relevant_chunks
        }

    def get_stats(self) -> Dict:
        return {
            'total_chunks': len(self.vector_store.chunks),
            'sources': list(set(chunk['source'] for chunk in self.vector_store.chunks))
        }

# Optional CLI
def main():
    import argparse
    parser = argparse.ArgumentParser(description="PDF RAG System")
    parser.add_argument('--upload', '-u', type=str, help='Upload a PDF file')
    parser.add_argument('--query', '-q', type=str, help='Ask a question')
    parser.add_argument('--stats', '-s', action='store_true', help='Show system stats')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    args = parser.parse_args()

    rag = RAGSystem()

    if args.upload:
        if os.path.exists(args.upload):
            chunks_added = rag.process_pdf(args.upload)
            print(f"Successfully processed PDF: {chunks_added} chunks added")
        else:
            print(f"File not found: {args.upload}")

    elif args.query:
        result = rag.query(args.query)
        print(f"\nQuestion: {args.query}")
        print(f"Answer: {result['answer']}")
        print("\nSources:")
        for i, source in enumerate(result['sources'][:3], 1):
            print(f"{i}. {source['source']} (Page {source['page_num']}) - Score: {source['similarity_score']:.3f}")

    elif args.stats:
        stats = rag.get_stats()
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Sources: {', '.join(stats['sources'])}")

    elif args.interactive:
        print("Interactive RAG System (type 'quit' to exit)")
        while True:
            query = input("\nEnter your question: ").strip()
            if query.lower() in ['quit', 'exit']:
                break
            result = rag.query(query)
            print(f"\nAnswer: {result['answer']}")
            print("\nTop sources:")
            for i, source in enumerate(result['sources'][:3], 1):
                print(f"{i}. {source['source']} (Page {source['page_num']}) - Score: {source['similarity_score']:.3f}")
    else:
        print("Use --help for usage instructions")


if __name__ == "__main__":
    main()
