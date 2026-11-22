from pathlib import Path
import glob
import pandas as pd


DATA_DIR = Path("data/raw")


def main():

    csv_files = glob.glob(str(DATA_DIR / "*.csv"))

    if not csv_files:

        print("No CSV files found in data/raw")
        return

    for file in csv_files:

        print("\n" + "=" * 60)

        print("FILE:", Path(file).name)

        df = pd.read_csv(
            file,
            nrows=5,
            low_memory=False
        )

        print("\nColumns:")

        print(list(df.columns))

        print("\nFirst five rows:")

        print(df.head())

        label_column = next(
            (
                c for c in df.columns
                if c.strip().lower() == "label"
            ),
            None
        )

        if label_column:

            print("\nLabel column found:", label_column)


if __name__ == "__main__":
    main()