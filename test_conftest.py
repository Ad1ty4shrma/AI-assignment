# tests/conftest.py
import pytest
import tempfile
import os
from pathlib import Path

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing"""
    return {
        'pages': [
            {
                'page_num': 1,
                'text': "This is the first page of a test document. It contains information about artificial intelligence and machine learning.",
                'source': 'sample.pdf'
            },
            {
                'page_num': 2,
                'text': "The second page discusses natural language processing and computer vision applications.",
                'source': 'sample.pdf'
            }
        ]
    }

@pytest.fixture
def sample_chunks():
    """Sample chunks for testing"""
    return [
        {
            'id': 0,
            'text': 'This is about artificial intelligence and machine learning.',
            'page_num': 1,
            'source': 'sample.pdf'
        },
        {
            'id': 1,
            'text': 'Natural language processing is a key area of AI research.',
            'page_num': 2,
            'source': 'sample.pdf'
        },
        {
            'id': 2,
            'text': 'Computer vision enables machines to interpret visual information.',
            'page_num': 2,
            'source': 'sample.pdf'
        }
    ]
