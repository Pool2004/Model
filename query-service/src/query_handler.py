import chromadb
import logging
from sentence_transformers import SentenceTransformer
from src.config import settings

logger = logging.getLogger(__name__)

class QueryHandler:
    def __init__(self):
        logger.info(f"Loading embedding model {settings.MODEL_NAME} for queries...")
        self.encoder = SentenceTransformer(settings.MODEL_NAME)
        
        try:
            self.chroma_client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
            self.collection = self.chroma_client.get_collection(name=settings.COLLECTION_NAME)
            logger.info("Connected to ChromaDB successfully")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            self.collection = None

    def search_context(self, query: str, n_results: int = 3):
        if not self.collection:
            logger.error("ChromaDB Collection is not initialized")
            return []
            
        try:
            query_vector = self.encoder.encode(query).tolist()
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=n_results
            )
            
            # format results
            context_documents = []
            if results and 'documents' in results and len(results['documents']) > 0:
                for idx, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][idx] if 'metadatas' in results and results['metadatas'] else {}
                    context_documents.append({
                        "text": doc,
                        "metadata": metadata,
                        "distance": results['distances'][0][idx] if 'distances' in results else 0.0
                    })
            return context_documents
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return []

    def generate_response(self, query: str, context_docs: list[dict]):
        # Mock de una llamada a LLM (ejemplo con OpenAI, Anthropic, LLaMA local, etc)
        if not context_docs:
            return "No se encontró información relevante para responder la pregunta."
        
        combined_context = "\n---\n".join([doc["text"] for doc in context_docs])
        
        fake_llm_response = (
            f"Basado en el contexto proporcionado ({len(context_docs)} documentos encontrados), "
            f"esta es una respuesta simulada a tu pregunta: '{query}'.\n\n"
            f"Contexto relevante:\n{combined_context}"
        )
        return fake_llm_response

query_instance = QueryHandler()
