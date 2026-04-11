from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    KAFKA_BROKER: str = "localhost:9092"
    KAFKA_TOPIC_RAW: str = "document-raw"
    KAFKA_TOPIC_EMBEDDINGS: str = "document-embeddings"
    KAFKA_GROUP_ID: str = "embedding-service-group"
    MODEL_NAME: str = "all-MiniLM-L6-v2"
    
    class Config:
        env_file = ".env"

settings = Settings()
