from pathlib import Path
import joblib
import pandas as pd


MODEL_PATH = Path("artifacts/traffic_model.joblib")


def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run train_model.py first."
        )

    return joblib.load(MODEL_PATH)


def get_features():

    artifact = load_model()

    return artifact["features"]


def predict_traffic(input_features):

    artifact = load_model()

    model = artifact["model"]

    required_features = artifact["features"]

    missing = [
        feature
        for feature in required_features
        if feature not in input_features
    ]

    extra = [
        feature
        for feature in input_features
        if feature not in required_features
    ]

    if missing or extra:

        raise ValueError(
            f"Missing features: {missing[:10]}. "
            f"Unexpected features: {extra[:10]}."
        )

    row = {}

    for feature in required_features:

        value = input_features[feature]

        try:
            row[feature] = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Feature {feature} must contain a number."
            )

    dataframe = pd.DataFrame(
        [row],
        columns=required_features
    )

    prediction = str(
        model.predict(dataframe)[0]
    )

    probabilities = model.predict_proba(dataframe)[0]

    confidence = float(max(probabilities))

    return prediction, confidence