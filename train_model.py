from pathlib import Path
import glob
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


DATA_DIR = Path("data/raw")
MODEL_DIR = Path("artifacts")

MODEL_DIR.mkdir(exist_ok=True)


def main():

    csv_files = glob.glob(str(DATA_DIR / "*.csv"))

    if not csv_files:
        raise SystemExit("No CSV files found in data/raw")

    dataframes = []

    for file in csv_files:

        print("Reading:", Path(file).name)

        df = pd.read_csv(file, low_memory=False)

        df.columns = [
            str(column).strip()
            for column in df.columns
        ]

        label_column = next(
            (
                column for column in df.columns
                if column.strip().lower() == "label"
            ),
            None
        )

        if label_column is None:
            print("Label column missing. Skipping.")
            continue

        # Use a manageable sample from each CSV.
        df = df.sample(
            n=min(len(df), 20000),
            random_state=42
        )

        df = df.rename(columns={label_column: "label"})

        dataframes.append(df)

    if not dataframes:
        raise SystemExit("No usable CSV files with a Label column.")

    data = pd.concat(
        dataframes,
        ignore_index=True
    )

    # Convert original labels into two classes.
    y = (
        data["label"]
        .astype(str)
        .str.strip()
        .str.upper()
        .apply(
            lambda value:
            "BENIGN" if value == "BENIGN" else "ATTACK"
        )
    )

    X = data.drop(columns=["label"])

    # Keep numeric features only.
    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [float("inf"), float("-inf")],
        float("nan")
    )

    # Remove columns without usable numeric values.
    X = X.dropna(axis=1, how="all")

    # Remove constant columns.
    X = X.loc[:, X.nunique(dropna=True) > 1]

    if X.empty:
        raise SystemExit("No usable numeric features found.")

    if y.nunique() < 2:
        raise SystemExit("Dataset needs both BENIGN and ATTACK records.")

    print("Dataset rows:", len(X))
    print("Number of features:", len(X.columns))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            )
        )
    ])

    print("Training model...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    joblib.dump(
        {
            "model": model,
            "features": list(X.columns)
        },
        MODEL_DIR / "traffic_model.joblib"
    )

    print("Model trained and saved successfully.")


if __name__ == "__main__":
    main()