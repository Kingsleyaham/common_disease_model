import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    API_VERSION: str = '/api/v1'
    PROJECT_NAME: str = "Disease Model"
    DEBUG: bool = False
    CORS_ORIGINS: list[str] = ['*']
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")


    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')


settings = Settings()