from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    QDRANT_URL: str = "localhost:6333"
    QDRANT_COLLECTION: str ="sample_collection"



settings = AppConfig()
