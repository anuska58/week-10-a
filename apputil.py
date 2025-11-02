import pandas as pd
import pickle
import numpy as np

# Load models
with open("model_1.pickle", "rb") as f:
    model_1 = pickle.load(f)

with open("model_2.pickle", "rb") as f:
    model_2 = pickle.load(f)

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

def predict_rating(df_X):
    """
    df_X = dataframe with columns ["100g_USD", "roast"]
    Returns an array of predicted rating values.
    """
    predictions = []

    for _, row in df_X.iterrows():
        usd = row["100g_USD"]
        roast = row["roast"]
        
        roast_value = roast_category(roast)

        if roast_value is None:   # Unknown or missing roast → use model_1
            pred = model_1.predict([[usd]])[0]
        else:                     # Known roast → use model_2
            pred = model_2.predict([[usd, roast_value]])[0]

        predictions.append(pred)

    return np.array(predictions)
 