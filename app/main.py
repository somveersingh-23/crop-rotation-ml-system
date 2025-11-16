"""
Production-ready FastAPI app for Render deployment with HuggingFace integration
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from huggingface_hub import hf_hub_download
import joblib
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Optional
import os

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Crop Rotation ML API",
    description="Crop recommendation system using ML models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models cache
models_cache = {}

def download_models_from_hf():
    """Download models from Hugging Face if not present"""
    logger.info("📥 Checking models...")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    repo_id = "somveersingh-23/crop-rotation-ml-system"
    files = ["ensemble_model.pkl", "scaler.pkl", "label_encoder.pkl"]
    
    for file in files:
        file_path = models_dir / file
        if not file_path.exists():
            logger.info(f"   Downloading {file} from HuggingFace...")
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=file,
                    local_dir=str(models_dir),
                    local_dir_use_symlinks=False
                )
                logger.info(f"   ✅ {file} downloaded")
            except Exception as e:
                logger.error(f"   ❌ Failed to download {file}: {e}")
                raise
        else:
            logger.info(f"   ✅ {file} already exists")
    
    logger.info("✅ All models ready")

def load_models():
    """Load models from models/ directory"""
    models_dir = Path("models")
    
    if not models_dir.exists():
        raise FileNotFoundError("Models directory not found!")
    
    try:
        logger.info("📦 Loading models into memory...")
        models_cache['ensemble'] = joblib.load(models_dir / 'ensemble_model.pkl')
        models_cache['scaler'] = joblib.load(models_dir / 'scaler.pkl')
        models_cache['label_encoder'] = joblib.load(models_dir / 'label_encoder.pkl')
        
        logger.info("✅ Models loaded successfully")
        logger.info(f"   Crops supported: {len(models_cache['label_encoder'].classes_)}")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        return False

# Load models on startup
@app.on_event("startup")
async def startup_event():
    """Download and load models when app starts"""
    logger.info("🚀 Starting application...")
    try:
        download_models_from_hf()
        success = load_models()
        if success:
            logger.info("🎉 Application ready!")
        else:
            logger.warning("⚠️  Models not loaded properly")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(models_cache) > 0,
        "models_count": len(models_cache)
    }

# Root endpoint
@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Crop Rotation ML API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "crops": "/crops",
            "docs": "/docs"
        }
    }

# Request model
class CropPredictionRequest(BaseModel):
    n: float = Field(..., ge=0, le=500, description="Nitrogen content")
    p: float = Field(..., ge=0, le=200, description="Phosphorus content")
    k: float = Field(..., ge=0, le=200, description="Potassium content")
    temperature: float = Field(..., ge=5, le=50, description="Temperature (°C)")
    humidity: float = Field(..., ge=0, le=100, description="Humidity (%)")
    ph: float = Field(..., ge=3.5, le=9.5, description="Soil pH")
    rainfall: float = Field(..., ge=0, le=400, description="Rainfall (mm)")

    class Config:
        json_schema_extra = {
            "example": {
                "n": 80,
                "p": 40,
                "k": 40,
                "temperature": 25,
                "humidity": 80,
                "ph": 6.5,
                "rainfall": 200
            }
        }

# Response model
class CropPrediction(BaseModel):
    crop: str
    confidence: float

class PredictionResponse(BaseModel):
    top_predictions: List[CropPrediction]
    recommended_crop: str
    confidence: float

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_crop(request: CropPredictionRequest):
    """Predict best crop based on soil and weather parameters"""
    
    if not models_cache:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Prepare features
        features = np.array([[
            request.n,
            request.p,
            request.k,
            request.temperature,
            request.humidity,
            request.ph,
            request.rainfall
        ]])
        
        # Scale features
        features_scaled = models_cache['scaler'].transform(features)
        
        # Predict
        probabilities = models_cache['ensemble'].predict_proba(features_scaled)[0]
        
        # Get top 5 predictions
        top_indices = np.argsort(probabilities)[-5:][::-1]
        
        predictions = []
        for idx in top_indices:
            crop = models_cache['label_encoder'].classes_[idx]
            confidence = float(probabilities[idx])
            predictions.append(CropPrediction(crop=crop, confidence=confidence))
        
        return PredictionResponse(
            top_predictions=predictions,
            recommended_crop=predictions[0].crop,
            confidence=predictions[0].confidence
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get supported crops
@app.get("/crops")
async def get_crops():
    """Get list of supported crops"""
    
    if not models_cache:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    crops = models_cache['label_encoder'].classes_.tolist()
    
    return {
        "total_crops": len(crops),
        "crops": sorted(crops)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
