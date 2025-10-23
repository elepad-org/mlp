"""FastAPI backend for MLP pattern recognition.

This API serves predictions from trained MLP models with proper
validation, error handling, and model information endpoints.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict
import numpy as np
from pathlib import Path

from mlp import MLP
from train import get_production_model, load_model_registry


# Initialize FastAPI app
app = FastAPI(
    title="MLP Pattern Recognition API",
    description="API for letter pattern recognition using Multilayer Perceptron",
    version="1.0.0",
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
_model: MLP | None = None


def get_model() -> MLP:
    """Get the loaded production model, lazy-loading if necessary."""
    global _model
    if _model is None:
        print("🔄 Loading production model...")
        _model = get_production_model()
        if _model is None:
            raise HTTPException(
                status_code=503,
                detail="No production model available. Please train a model first."
            )
        print("✅ Model loaded successfully")
    return _model


# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    """Request model for predictions."""
    pattern: List[int] = Field(
        ...,
        description="10x10 pattern represented as 100 integers (0 or 1)",
        min_length=100,
        max_length=100,
    )

    @field_validator('pattern')
    @classmethod
    def validate_pattern(cls, v):
        """Validate that pattern contains only 0s and 1s."""
        if not all(x in [0, 1] for x in v):
            raise ValueError("Pattern must contain only 0s and 1s")
        return v


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    letter: str = Field(..., description="Predicted letter (b, d, or f)")
    probabilities: Dict[str, float] = Field(..., description="Probability for each class")
    confidence: float = Field(..., description="Confidence score (max probability)")


class ModelInfo(BaseModel):
    """Response model for model information."""
    version: str
    accuracy: float
    created_at: str
    is_production: bool
    hyperparameters: Dict
    architecture: Dict


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    message: str


# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "MLP Pattern Recognition API",
        "version": "1.0.0",
        "docs": "/docs",
        "model_info": "/model/info",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        model = get_model()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            message="API is running and model is loaded"
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            message=f"API is running but model is not available: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict the letter from a 10x10 pattern.
    
    - **pattern**: List of 100 integers (0 or 1) representing the 10x10 grid
    
    Returns the predicted letter and confidence scores.
    """
    try:
        model = get_model()
        
        # Convert to numpy array
        pattern_array = np.array(request.pattern, dtype=np.float64)
        
        # Get prediction
        predicted_letter = model.classify(pattern_array)
        probabilities = model.predict_proba(pattern_array)
        confidence = max(probabilities.values())
        
        return PredictionResponse(
            letter=predicted_letter,
            probabilities=probabilities,
            confidence=confidence,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """
    Get information about the current production model.
    
    Returns model version, accuracy, hyperparameters, and architecture details.
    """
    try:
        # Get model info from registry
        registry = load_model_registry()
        production_model = None
        
        for model_entry in registry["models"]:
            if model_entry.get("is_production", False):
                production_model = model_entry
                break
        
        if production_model is None:
            raise HTTPException(
                status_code=404,
                detail="No production model found in registry"
            )
        
        # Get loaded model for architecture info
        model = get_model()
        
        return ModelInfo(
            version=production_model["version"],
            accuracy=production_model["accuracy"],
            created_at=production_model["created_at"],
            is_production=True,
            hyperparameters=production_model["hyperparameters"],
            architecture=model.metadata["architecture"],
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )


@app.get("/models/list")
async def list_models():
    """
    List all models in the registry.
    
    Returns information about all trained models.
    """
    try:
        registry = load_model_registry()
        return {
            "models": registry.get("models", []),
            "total": len(registry.get("models", [])),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list models: {str(e)}"
        )


# Run with: uvicorn api:app --reload --host 0.0.0.0 --port 8000
# For Render deployment, it will use PORT environment variable
if __name__ == "__main__":
    import uvicorn
    import os
    
    # Render sets PORT env var, default to 8000 for local development
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        app, 
        host="0.0.0.0",  # Required by Render
        port=port
    )
