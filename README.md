# Cybersecurity Threat Detection System

## About the Project

This project uses Machine Learning to classify network traffic
into two categories:

1. BENIGN
2. ATTACK

Dataset: CIC-IDS2017

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Random Forest
- FastAPI
- SQLite
- SQLAlchemy
- Streamlit

## Project Workflow

1. Read network traffic CSV files.
2. Explore and prepare the dataset.
3. Train a Random Forest classifier.
4. Save the trained model.
5. Receive network features through FastAPI.
6. Predict BENIGN or ATTACK.
7. Save predictions in SQLite.
8. Display results in Streamlit.

## Installation

Install dependencies:

python -m pip install -r requirements.txt

## Explore Dataset

python explore_data.py

## Train Model

python train_model.py

## Run API

python -m uvicorn main:app --reload

## API Documentation

http://127.0.0.1:8000/docs

## Run Dashboard

Open another terminal:

python -m streamlit run dashboard.py

## Limitations

This is an educational baseline project.
It is not a production-ready intrusion detection system.

The model groups all non-BENIGN labels into ATTACK.
Its predictions must be evaluated using appropriate test data.