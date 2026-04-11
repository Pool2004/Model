import logging
from sentence_transformers import SentenceTransformer
from src.config import settings

logger = logging.getLogger(__name__)

class DocumentEmbedder:
    def __init__(self, model_name: str = settings.MODEL_NAME):
        logger.info(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        logger.info("Model loaded successfully.")

    def embed_text(self, text: str) -> list[float]:
        try:
            embeddings = self.model.encode(text)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

embedder_instance = DocumentEmbedder()
