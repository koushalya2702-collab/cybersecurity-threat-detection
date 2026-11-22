from pydantic import BaseModel, Field
from typing import Dict


class TrafficInput(BaseModel):

    features: Dict[str, float] = Field(
        ...,
        min_length=1
    )


class PredictionOutput(BaseModel):

    id: int

    prediction: str

    confidence: float

    message: str