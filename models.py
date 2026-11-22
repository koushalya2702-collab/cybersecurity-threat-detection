from sqlalchemy import Column, Integer, String, Float, JSON
from database import Base


class Prediction(Base):

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    features = Column(JSON, nullable=False)

    prediction = Column(String, nullable=False)

    confidence = Column(Float, nullable=False)