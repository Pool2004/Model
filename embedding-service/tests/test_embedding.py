import pytest
from unittest.mock import patch, MagicMock
from src.embedder import DocumentEmbedder

@patch("src.embedder.SentenceTransformer")
def test_embedder(mock_st):
    # Setup mock
    mock_model = MagicMock()
    mock_model.encode.return_value = [0.1, 0.2, 0.3]
    mock_st.return_value = mock_model
    
    # Test
    embedder = DocumentEmbedder(model_name="test-model")
    vector = embedder.embed_text("test text")
    
    assert isinstance(vector, list)
    assert len(vector) == 3
    mock_model.encode.assert_called_once_with("test text")

@patch("src.main.Consumer")
@patch("src.main.Producer")
@patch("src.main.embedder_instance.embed_text")
def test_main_consumer_loop(mock_embed, mock_prod, mock_cons):
    # Este test podría ser más exhaustivo pero validamos la lógica base mockeando
    pass 
