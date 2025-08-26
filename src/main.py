from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from inference import predict_price, batch_predict as inference_batch_predict
from schemas import HousePredictionRequest, PredictionResponse
from typing import List

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Prediction API",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health", response_model=dict)
async def health_check():
    return {"status": "healthy", "model_loaded": True}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: HousePredictionRequest):
    return predict_price(request)

# Batch prediction endpoint
@app.post("/batch-predict", response_model=List[PredictionResponse])
async def batch_predict_endpoint(requests: List[HousePredictionRequest]):
    return inference_batch_predict(requests)
