import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import pickle

url="https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"

# df=pd.read_csv(url)
# df=df.dropna()
# print(df.info())
def train_model_1():
    # Load dataset
    df = pd.read_csv(url)

    # Define features and target variable
    X = df[["100g_USD"]]
    y= df["rating"]

    # Train linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Save the trained model to a file
    with open("model1.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model trained and saved as model1.pkl")

def roast_category(value):
    mapping = {
        "Light":0,
        "Medium-Light":1,
        "Medium":2,
        "Medium-Dark":3,
        "Dark":4
    }
    return mapping.get(value, None)

def train_model_2():
    # Load dataset
    df = pd.read_csv(url)

    df["roast_cat"]=df["roast"].apply(roast_category)
    # Define features and target variable
    X = df[["100g_USD", "roast_cat"]]
    y = df["rating"]

    # Train decision tree regression model
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)

    # Save the trained model to a file
    with open("model2.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model trained and saved as model2.pkl")
    
if __name__ == "__main__":
    train_model_1()
    train_model_2()
