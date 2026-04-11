import pytest
from unittest.mock import patch, MagicMock

@patch("src.chroma_client.chromadb.HttpClient")
def test_chroma_storage(mock_http_client):
    mock_client_instance = MagicMock()
    mock_collection = MagicMock()
    mock_client_instance.get_or_create_collection.return_value = mock_collection
    mock_http_client.return_value = mock_client_instance
    
    from src.chroma_client import ChromaStorage
    storage = ChromaStorage()
    
    storage.store_document(
        doc_id="test_id",
        text="test content",
        vector=[0.1, 0.2],
        metadata={"source": "test"}
    )
    
    mock_collection.add.assert_called_once_with(
        ids=["test_id"],
        embeddings=[[0.1, 0.2]],
        documents=["test content"],
        metadatas=[{"source": "test"}]
    )

@patch("src.main.Consumer")
@patch("src.main.chroma_instance.store_document")
def test_main_consumer_loop(mock_store, mock_cons):
    # Setup mock validation logic
    pass
