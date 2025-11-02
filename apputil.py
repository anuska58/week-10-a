import pandas as pd
import pickle
import numpy as np

# Same roast mapping used in train.py
def roast_category(value):
    mapping = {
        "Light": 0,
        "Medium-Light": 1,
        "Medium": 2,
        "Medium-Dark": 3,
        "Dark": 4
    }
    return mapping.get(value, None)

# Lazy loading: models are loaded only once when needed
model_1 = None
model_2 = None

def load_models():
    global model_1, model_2
    if model_1 is None:
        try:
            with open("model_1.pickle", "rb") as f:
                model_1 = pickle.load(f)
        except FileNotFoundError:
            print("model_1.pickle not found")
    if model_2 is None:
        try:
            with open("model_2.pickle", "rb") as f:
                model_2 = pickle.load(f)
        except FileNotFoundError:
            print("model_2.pickle not found")

def predict_rating(df_X):
    """
    df_X = DataFrame with columns ["100g_USD", "roast"]
    Returns an array of predicted rating values.
    """
    load_models()  # Ensure models are available

    predictions = []

    for _, row in df_X.iterrows():
        usd = row["100g_USD"]
        roast = row["roast"]
        
        roast_value = roast_category(roast)

        if roast_value is None or model_2 is None:  # Use model_1 if roast unknown or model_2 missing
            pred = model_1.predict([[usd]])[0]
        else:
            pred = model_2.predict([[usd, roast_value]])[0]

        predictions.append(pred)

    return np.array(predictions)
