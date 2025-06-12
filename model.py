import pandas as pd
from sklearn.linear_model import LinearRegression

def load_and_process_data(csv_path):
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()

    df = df.dropna()
    return df

def train_model(df):
    features = ['Open', 'High', 'Low', 'Volume', 'MA5', 'MA10']
    target = 'Close'

    X = df[features]
    y = df[target]

    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_price(model, df):
    features = ['Open', 'High', 'Low', 'Volume', 'MA5', 'MA10']
    return model.predict(df[features])
