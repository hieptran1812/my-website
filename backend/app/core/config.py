from pydantic_settings import BaseSettings
from typing import List, Optional
from pathlib import Path


class Settings(BaseSettings):
    PROJECT_NAME: str = "Hiep Tran Portfolio API"
    PROJECT_DESCRIPTION: str = "Backend API for Hiep Tran's personal portfolio website"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    
    # Security
    SECRET_KEY: str = "your-super-secret-key-please-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    ALGORITHM: str = "HS256"

    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/portfolio"

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "https://hieptran.dev", "http://127.0.0.1:3000"]

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    DATA_DIR: Path = BASE_DIR / "data"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # Default admin user
    DEFAULT_ADMIN_USERNAME: str = "hieptran1812"
    DEFAULT_ADMIN_EMAIL: str = "hieptran.jobs@gmail.com"
    DEFAULT_ADMIN_FULL_NAME: str = "Hiep Tran"
    DEFAULT_ADMIN_PASSWORD: str = "adminpassword"

    # Email configuration (for contact form)
    SMTP_TLS: bool = True
    SMTP_PORT: Optional[int] = None
    SMTP_HOST: Optional[str] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAILS_FROM_EMAIL: Optional[str] = None
    EMAILS_FROM_NAME: Optional[str] = None

    # Cloud Storage (AWS S3)
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_S3_BUCKET_NAME: Optional[str] = None
    AWS_S3_REGION: Optional[str] = "us-east-1"


    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()

# Ensure necessary directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
