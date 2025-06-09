import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GEMINI_API_KEY: str
    MAX_RETRIES: int = 2
    DEFAULT_MODEL: str = "gemini-1.5-flash"
    
    class Config:
        env_file = ".env"

config = Settings()