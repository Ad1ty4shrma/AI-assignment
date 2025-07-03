# tests/test_rag_system.py
import pytest
import tempfile
import os
from pathlib import Path
import numpy as np
from unittest.mock import Mock, patch

from rag_system import (
    PDFProcessor, 
    EmbeddingGenerator, 
    VectorStore, 
    LLMClient, 
    RAGSystem
)

class TestPDFProcessor:
    
    def test_chunk_text_basic(self):
        processor = PDFProcessor(chunk_size=50, overlap=10)
        
        pages_text = [{
            'page_num': 1,
            'text': "This is a test document. It has multiple sentences. We want to test chunking.",
            'source': 'test.pdf'
        }]
        
        chunks = processor.chunk_text(pages_text)
        
        assert len(chunks) > 0
        assert all('text' in chunk for chunk in chunks)
        assert all('page_num' in chunk for chunk in chunks)
        assert all('source' in chunk for chunk in chunks)
    
    def test_chunk_text_empty(self):
        processor = PDFProcessor()
        
        pages_text = [{
            'page_num': 1,
            'text': "",
            'source': 'empty.pdf'
        }]
        
        chunks = processor.chunk_text(pages_text)
        assert len(chunks) == 0

class TestEmbeddingGenerator:
    
    @pytest.fixture
    def embedding_generator(self):
        return EmbeddingGenerator("all-MiniLM-L6-v2")
    
    def test_generate_single_embedding(self, embedding_generator):
        text = "This is a test sentence."
        embedding = embedding_generator.generate_single_embedding(text)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == embedding_generator.dimension
        assert embedding.dtype == np.float32
    
    def test_generate_multiple_embeddings(self, embedding_generator):
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = embedding_generator.generate_embeddings(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, embedding_generator.dimension)
        assert embeddings.dtype == np.float32

class TestVectorStore:
    
    @pytest.fixture
    def vector_store(self):
        return VectorStore(dimension=384)  # all-MiniLM-L6-v2 dimension
    
    def test_add_and_search_embeddings(self, vector_store):
        # Create test data
        chunks = [
            {'id': 0, 'text': 'This is about cats', 'page_num': 1, 'source': 'test.pdf'},
            {'id': 1, 'text': 'This is about dogs', 'page_num': 2, 'source': 'test.pdf'},
            {'id': 2, 'text': 'This is about birds', 'page_num': 3, 'source': 'test.pdf'}
        ]
        
        embeddings = np.random.rand(3, 384).astype(np.float32)
        
        # Add embeddings
        vector_store.add_embeddings(embeddings, chunks)
        
        # Search
        query_embedding = np.random.rand(384).astype(np.float32)
        results = vector_store.search(query_embedding, k=2)
        
        assert len(results) == 2
        assert all('similarity_score' in result for result in results)
        assert all('text' in result for result in results)
    
    def test_save_and_load(self, vector_store):
        # Create test data
        chunks = [{'id': 0, 'text': 'Test chunk', 'page_num': 1, 'source': 'test.pdf'}]
        embeddings = np.random.rand(1, 384).astype(np.float32)
        
        vector_store.add_embeddings(embeddings, chunks)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            vector_store.save(tmp.name)
            
            # Create new store and load
            new_store = VectorStore(dimension=384)
            new_store.load(tmp.name)
            
            assert len(new_store.chunks) == 1
            assert new_store.chunks[0]['text'] == 'Test chunk'
            
            # Clean up
            os.unlink(f"{tmp.name}.index")
            os.unlink(f"{tmp.name}.chunks")

class TestLLMClient:
    
    @pytest.fixture
    def llm_client(self):
        return LLMClient()
    
    @patch('requests.post')
    def test_generate_answer_success(self, mock_post, llm_client):
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'response': 'This is a test answer.'}
        mock_post.return_value = mock_response
        
        chunks = [
            {'text': 'Test context', 'source': 'test.pdf', 'page_num': 1}
        ]
        
        answer = llm_client.generate_answer("Test question?", chunks)
        
        assert answer == "This is a test answer."
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_generate_answer_error(self, mock_post, llm_client):
        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        chunks = [
            {'text': 'Test context', 'source': 'test.pdf', 'page_num': 1}
        ]
        
        answer = llm_client.generate_answer("Test question?", chunks)
        
        assert "Error: Unable to generate response" in answer

class TestRAGSystem:
    
    @pytest.fixture
    def rag_system(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield RAGSystem(storage_dir=temp_dir)
    
    def test_initialization(self, rag_system):
        assert rag_system.pdf_processor is not None
        assert rag_system.embedding_generator is not None
        assert rag_system.vector_store is not None
        assert rag_system.llm_client is not None
    
    def test_get_stats_empty(self, rag_system):
        stats = rag_system.get_stats()
        
        assert stats['total_chunks'] == 0
        assert stats['sources'] == []
    
    def test_query_empty_system(self, rag_system):
        result = rag_system.query("What is this about?")
        
        assert "No documents have been processed" in result['answer']
        assert result['sources'] == []
    
    @patch('rag_system.fitz.open')
    def test_process_pdf_mock(self, mock_fitz_open, rag_system):
        # Mock PDF document
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "This is test content from a PDF document."
        mock_doc.load_page.return_value = mock_page
        mock_doc.__len__.return_value = 1
        mock_fitz_open.return_value = mock_doc
        
        # Process the "PDF"
        chunks_added = rag_system.process_pdf("fake.pdf")
        
        assert chunks_added > 0
        assert len(rag_system.vector_store.chunks) > 0
        
        # Test stats
        stats = rag_system.get_stats()
        assert stats['total_chunks'] > 0
        assert 'fake.pdf' in stats['sources']


class TestIntegration:
    
    @pytest.fixture
    def integration_system(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield RAGSystem(storage_dir=temp_dir)
    
    def test_full_pipeline_mock(self, integration_system):
        # Mock all external dependencies
        with patch('rag_system.fitz.open') as mock_fitz, \
             patch('requests.post') as mock_post:
            
            # Mock PDF
            mock_doc = Mock()
            mock_page = Mock()
            mock_page.get_text.return_value = "This document discusses artificial intelligence and machine learning."
            mock_doc.load_page.return_value = mock_page
            mock_doc.__len__.return_value = 1
            mock_fitz.return_value = mock_doc
            
            # Mock LLM response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'response': 'The document discusses AI and ML technologies.'
            }
            mock_post.return_value = mock_response
            
            # Test full pipeline
            chunks_added = integration_system.process_pdf("test.pdf")
            assert chunks_added > 0
            
            result = integration_system.query("What does the document discuss?")
            assert "AI and ML" in result['answer']
            assert len(result['sources']) > 0
