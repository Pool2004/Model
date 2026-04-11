from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    KAFKA_BROKER: str = "localhost:9092"
    KAFKA_TOPIC_RAW: str = "document-raw"
    
    class Config:
        env_file = ".env"

settings = Settings()
