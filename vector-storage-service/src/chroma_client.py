import chromadb
from chromadb.config import Settings as ChromaSettings
import logging
from src.config import settings

logger = logging.getLogger(__name__)

class ChromaStorage:
    def __init__(self):
        try:
            logger.info(f"Connecting to ChromaDB at {settings.CHROMA_HOST}:{settings.CHROMA_PORT}")
            self.client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
            self.collection = self.client.get_or_create_collection(name=settings.COLLECTION_NAME)
            logger.info(f"Connected to Chroma collection '{settings.COLLECTION_NAME}'")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise

    def store_document(self, doc_id: str, text: str, vector: list[float], metadata: dict):
        try:
            # Metadata requires primitive values in Chroma
            clean_metadata = {k: str(v) for k, v in metadata.items()} if metadata else {}
            
            self.collection.add(
                ids=[doc_id],
                embeddings=[vector],
                documents=[text],
                metadatas=[clean_metadata] if clean_metadata else None
            )
            logger.info(f"Stored document {doc_id} in ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Error storing document {doc_id} in ChromaDB: {e}")
            return False

chroma_instance = ChromaStorage()
