from sqlalchemy.orm import Session
from models import Prediction


def save_prediction(
    db: Session,
    features: dict,
    prediction: str,
    confidence: float
):

    record = Prediction(
        features=features,
        prediction=prediction,
        confidence=confidence
    )

    db.add(record)

    db.commit()

    db.refresh(record)

    return record


def get_all_predictions(db: Session):

    return (
        db.query(Prediction)
        .order_by(Prediction.id.desc())
        .limit(100)
        .all()
    )