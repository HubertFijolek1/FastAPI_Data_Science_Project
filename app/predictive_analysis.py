import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def train_predictive_model(df: pd.DataFrame, features, target):
    """
    Trains a basic LinearRegression on given features -> target.
    """
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict(model, new_data: pd.DataFrame):
    """
    Predict on new_data using the trained model.
    """
    return model.predict(new_data)
