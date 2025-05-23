# app/core/config.py
from pydantic_settings import BaseSettings
from pydantic import field_validator, computed_field
from typing import Optional, Dict, Any, List, Union
import os
from pathlib import Path
import logging

class Settings(BaseSettings):
    """Application settings with development-friendly defaults."""
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "IoMT Attack Detection System"
    VERSION: str = "1.0.0"
    
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # CORS settings - restrictive for production, permissive for development
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # React dev server
        "http://localhost:8080",  # Vue dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
    ] if os.getenv("ENVIRONMENT", "development") == "production" else ["*"]
    
    # Security
    SECRET_KEY: str = os.getenv(
        "SECRET_KEY", 
        "dev-secret-key-change-in-production-12345"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))  # 8 hours
    
    # Database settings (for future use)
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
    
    # Model settings with validation
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/lightgbm_model.pkl")
    FEATURE_PROCESSOR_PATH: str = os.getenv("FEATURE_PROCESSOR_PATH", "models/feature_mapping.pkl")
    SCALER_PATH: str = os.getenv("SCALER_PATH", "models/scaler.pkl")
    
    # Create models directory if it doesn't exist
    @field_validator('MODEL_PATH', 'FEATURE_PROCESSOR_PATH', 'SCALER_PATH')
    @classmethod
    def create_model_dirs(cls, v):
        model_dir = Path(v).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        return v
    
    # Alert settings
    ALERT_THRESHOLD: float = float(os.getenv("ALERT_THRESHOLD", "0.7"))
    MAX_ALERTS_PER_MINUTE: int = int(os.getenv("MAX_ALERTS_PER_MINUTE", "10"))
    
    # Dashboard settings
    MAX_DASHBOARD_DATAPOINTS: int = int(os.getenv("MAX_DASHBOARD_DATAPOINTS", "1000"))
    DASHBOARD_REFRESH_INTERVAL: int = int(os.getenv("DASHBOARD_REFRESH_INTERVAL", "30"))  # seconds
    
    # Logging settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG" if DEBUG else "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE", "logs/app.log")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_MAX_SIZE: int = int(os.getenv("LOG_MAX_SIZE", "10485760"))  # 10MB
    LOG_BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", "5"))
    
    # Create logs directory if it doesn't exist
    @field_validator('LOG_FILE')
    @classmethod
    def create_log_dir(cls, v):
        if v:
            log_dir = Path(v).parent
            log_dir.mkdir(parents=True, exist_ok=True)
        return v
    
    # Data processing settings
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    MAX_PAYLOAD_SIZE: int = int(os.getenv("MAX_PAYLOAD_SIZE", "16777216"))  # 16MB
    
    # Monitoring settings
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    METRICS_PORT: int = int(os.getenv("METRICS_PORT", "9090"))
    
    # Development settings
    RELOAD: bool = DEBUG  # Auto-reload in development
    MOCK_ML_MODELS: bool = os.getenv("MOCK_ML_MODELS", "false").lower() == "true"
    
    # Validation
    @field_validator('ALERT_THRESHOLD')
    @classmethod
    def validate_alert_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('ALERT_THRESHOLD must be between 0 and 1')
        return v
    
    @field_validator('SECRET_KEY')
    @classmethod
    def validate_secret_key(cls, v, info):
        if info.data.get('ENVIRONMENT') == 'production' and v == 'dev-secret-key-change-in-production-12345':
            raise ValueError('Must set SECRET_KEY in production environment')
        return v
    
    # Property methods for common paths
    @computed_field
    @property
    def models_dir(self) -> Path:
        return Path(self.MODEL_PATH).parent
    
    @computed_field
    @property
    def logs_dir(self) -> Path:
        return Path(self.LOG_FILE).parent if self.LOG_FILE else Path("logs")
    
    @computed_field
    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"
    
    @computed_field
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"
    
    # Method to get logging configuration
    def get_logging_config(self) -> Dict[str, Any]:
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'default': {
                    'format': self.LOG_FORMAT,
                },
                'detailed': {
                    'format': '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'level': self.LOG_LEVEL,
                    'class': 'logging.StreamHandler',
                    'formatter': 'default',
                },
                'file': {
                    'level': self.LOG_LEVEL,
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': self.LOG_FILE,
                    'maxBytes': self.LOG_MAX_SIZE,
                    'backupCount': self.LOG_BACKUP_COUNT,
                    'formatter': 'detailed' if self.is_production else 'default',
                },
            },
            'loggers': {
                'iomt_detection': {
                    'level': self.LOG_LEVEL,
                    'handlers': ['console', 'file'] if self.LOG_FILE else ['console'],
                    'propagate': False,
                },
            },
            'root': {
                'level': self.LOG_LEVEL,
                'handlers': ['console'],
            }
        }
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"
        
    def display_config(self) -> str:
        """Display current configuration (hiding sensitive data)."""
        config_display = []
        for field_name, field_value in self.__dict__.items():
            if 'SECRET' in field_name.upper() or 'PASSWORD' in field_name.upper():
                display_value = '***HIDDEN***'
            else:
                display_value = field_value
            config_display.append(f"{field_name}: {display_value}")
        return "\n".join(config_display)

# Create global settings instance
settings = Settings()

# Convenience function for logging setup
def setup_logging():
    """Setup logging configuration."""
    import logging.config
    logging.config.dictConfig(settings.get_logging_config())
    logger = logging.getLogger('iomt_detection')
    logger.info(f"Logging configured for {settings.ENVIRONMENT} environment")
    return logger