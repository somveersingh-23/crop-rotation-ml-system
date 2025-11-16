"""
Configuration settings for production
"""
import os
from typing import List
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Crop Rotation AI API"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # OpenWeather API
    OPENWEATHER_API_KEY: str = "2920d4cb32f1e61e2ee1b27aae2cfe03"
    OPENWEATHER_BASE_URL: str = "https://api.openweathermap.org/data/2.5"
    
    # Model paths
    RF_MODEL_PATH: str = "models/rf_model.pkl"
    XGB_MODEL_PATH: str = "models/xgb_model.pkl"
    SCALER_PATH: str = "models/scaler.pkl"
    LABEL_ENCODER_PATH: str = "models/label_encoder.pkl"
    METADATA_PATH: str = "models/metadata.json"
    
    # Data paths
    CROP_DATA_PATH: str = "data/raw/crop_recommendation.csv"
    ROTATION_RULES_PATH: str = "data/raw/crop_rotation_rules.csv"
    MARKET_PRICES_PATH: str = "data/raw/market_prices.csv"
    SOIL_MAPPING_PATH: str = "data/raw/soil_crop_mapping.csv"
    
    # ML Settings
    CONFIDENCE_THRESHOLD: float = 0.75
    MAX_RECOMMENDATIONS: int = 4
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
