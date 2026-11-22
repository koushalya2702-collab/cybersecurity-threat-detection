
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from schemas import TrafficInput, PredictionOutput
from ml_model import predict_traffic
from crud import save_prediction

# Create database tables
Base.metadata.create_all(bind=engine)

# Create FastAPI application
app = FastAPI(
    title="Cybersecurity Threat Detection API",
    version="1.0.0"
)

# Allow dashboard to communicate with API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def home():
    return {
        "message": "Cybersecurity Threat Detection API is running"
    }


@app.get("/status")
def check_status():
    return {
        "status": "Active",
        "project": "Cybersecurity Threat Detection"
    }


@app.post("/analyze", response_model=PredictionOutput)
@app.post("/predict", response_model=PredictionOutput)
def analyze_traffic(
    traffic: TrafficInput,
    db: Session = Depends(get_db)
):
    try:
        # Get prediction from the trained ML model
        prediction, confidence = predict_traffic(
            traffic.features
        )

        # Save prediction in the database
        record = save_prediction(
            db=db,
            features=traffic.features,
            prediction=prediction,
            confidence=confidence
        )

        # Return prediction result
        return {
            "id": record.id,
            "prediction": prediction,
            "confidence": confidence,
            "message": "Traffic analysis completed successfully"
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
