import joblib
from sklearn.ensemble import RandomForestClassifier

def train_model(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, "models/model.pkl")
    return "models/model.pkl"