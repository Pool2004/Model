from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    KAFKA_BROKER: str = "localhost:9092"
    KAFKA_TOPIC_EMBEDDINGS: str = "document-embeddings"
    KAFKA_GROUP_ID: str = "vector-storage-group"
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    COLLECTION_NAME: str = "rag_documents"
    
    class Config:
        env_file = ".env"

settings = Settings()
