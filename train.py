import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

url="https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"

# df=pd.read_csv(url)
# df=df.dropna()
# print(df.info())
def main():
    # Load dataset
    df = pd.read_csv(url)

    # Preprocess data: drop rows with missing values
    df = df.dropna()

    # Define features and target variable
    X = df[["100g_USD"]]
    y= df["rating"]

    # Train linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Save the trained model to a file
    with open("linear_regression_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model trained and saved as linear_regression_model.pkl")

if __name__ == "__main__":
    main()
