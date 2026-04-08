from fastapi import FastAPI
from pydantic import BaseModel
from src.app.model.inference import predict

# Initialize FastAPI application
app = FastAPI(
    title="Churn Prediction API",
    description="ML API for predicting customer churn",
    version="1.0.0"
)

# === HEALTH CHECK ENDPOINT ===
@app.get("/")
def root():
    """
    Health check endpoint.
    """
    return {"status": "ok"}


# === REQUEST DATA SCHEMA ===
class InputData(BaseModel):
    avg_days_between: float
    has_multiple_purchases: int
    Recency: int
    Frequency: int
    Monetary: float
    returns: int


@app.post("/predict")
def get_prediction(data: InputData):
    """
    Main prediction endpoint for customer churn prediction.

    """
    try:
        # Convert Pydantic model to dict and call inference pipeline
        result = predict(data.dict())
        return {"prediction": result}
    except Exception as e:
        # Return error details for debugging (consider logging in production)
        return {"error": str(e)}