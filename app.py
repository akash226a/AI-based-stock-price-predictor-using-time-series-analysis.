import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objs as go

app = Flask(__name__)

def get_suggestion(actual, predicted):
    diff_pct = ((predicted - actual) / actual) * 100
    if diff_pct > 3:
        return 'BUY'
    elif diff_pct < -3:
        return 'SELL'
    else:
        return 'HOLD'

def predict_prices(df):
    df = df.copy()
    df = df[['Open', 'Close']].dropna()

    df['Actual Price'] = df['Close']
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    X = df[['Open']]
    y = df['Target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    # Predictions for test set (for metrics)
    y_pred = model.predict(X_test)

    # Metrics calculation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'MAE': round(mae, 2),
        'MSE': round(mse, 2),
        'RMSE': round(rmse, 2),
        'R2 Score': round(r2, 2)
    }

    # Predict full range
    df['Predicted Price'] = model.predict(scaler.transform(df[['Open']]))
    y_pred = model.predict(X_test)
    

 

    # Suggestion per row
    df['Suggestion'] = df.apply(lambda row: get_suggestion(row['Actual Price'], row['Predicted Price']), axis=1)

    # Future prediction
    last_open = df.iloc[-1]['Open']
    last_open_scaled = scaler.transform([[last_open]])
    future_price = model.predict(last_open_scaled)[0]
    latest_actual = df.iloc[-1]['Actual Price']
    future_suggestion = get_suggestion(latest_actual, future_price)

    future_result = {
        'Future Price': round(future_price, 2),
        'Suggestion': future_suggestion
    }

    return df[['Open', 'Close', 'Actual Price', 'Predicted Price', 'Suggestion']], future_result, metrics

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filename = file.filename  # <- Capture filename
            df = pd.read_csv(file)
            df, future , metrics = predict_prices(df)
            df = df.round(2)
            recent_data = df.tail(10).to_dict(orient='records')

            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=df['Actual Price'], name='Actual Price', line=dict(color='blue')))
            fig.add_trace(go.Scatter(y=df['Predicted Price'], name='Predicted Price', line=dict(color='orange')))
            fig.update_layout(title='Actual vs Predicted Price',
                              xaxis_title='Days', yaxis_title='Price')
            graph = fig.to_html(full_html=False)

            return render_template('dashboard.html', data=recent_data, plot=graph, future=future, filename=filename, metrics = metrics)

    return render_template('dashboard.html', data=None, plot=None, future=None, filename=None)

if __name__ == '__main__':
    app.run(debug=True)