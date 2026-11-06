import os
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App Information
    APP_NAME: str = "Virtual Try-On E-Commerce API"
    APP_VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = ["*"]

    # Security & API Key Auth
    API_KEY_SECRET: str = "vton_dev_secret_key_change_in_production"
    REQUIRE_API_KEY: bool = False

    # Redis & Task Queue
    USE_CELERY: bool = os.getenv("USE_CELERY", "false").lower() in ("true", "1", "yes")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

    # Storage Backend: "local" or "s3" (Cloudflare R2 / AWS S3)
    STORAGE_BACKEND: Literal["local", "s3"] = "local"
    LOCAL_STORAGE_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage_uploads")
    
    # S3 / Cloudflare R2 Credentials
    S3_ENDPOINT_URL: str = os.getenv("S3_ENDPOINT_URL", "")
    S3_ACCESS_KEY_ID: str = os.getenv("S3_ACCESS_KEY_ID", "")
    S3_SECRET_ACCESS_KEY: str = os.getenv("S3_SECRET_ACCESS_KEY", "")
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "vton-renders")
    S3_PUBLIC_DOMAIN: str = os.getenv("S3_PUBLIC_DOMAIN", "https://cdn.tryon.ai")

    # Engine & Models
    CHECKPOINT_DIR: str = os.getenv("CHECKPOINT_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "engine", "weights"))
    DEFAULT_TRYON_MODEL: Literal["leffa", "idm_vton", "cat_vton"] = "leffa"
    DEVICE: str = "cuda" if os.getenv("FORCE_CUDA", "0") == "1" else "cpu"
    MAX_IMAGE_DIMENSION: int = 1024

settings = Settings()

