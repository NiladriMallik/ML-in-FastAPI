import joblib
import pandas as pd

def make_prediction(df: pd.DataFrame):
    model = joblib.load("models/model.pkl")
    predictions = model.predict(df)
    return predictions.tolist()