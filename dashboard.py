import pandas as pd
import plotly.graph_objs as go
from ta.trend import MACD
from ta.momentum import RSIIndicator

def plot_prediction(actual, predicted):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=actual.index, y=actual, mode='lines+markers',
                             name='Actual Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=predicted.index, y=predicted, mode='lines+markers',
                             name='Predicted Price', line=dict(color='orange')))

    threshold = 0.03
    actions = []
    for i in range(len(actual)):
        diff = (predicted.iloc[i] - actual.iloc[i]) / actual.iloc[i]
        if diff > threshold:
            actions.append('Buy')
        elif diff < -threshold:
            actions.append('Sell')
        else:
            actions.append('Hold')

    fig.add_trace(go.Scatter(x=actual.index, y=actual,
                             mode='markers',
                             marker=dict(color=['green' if a == 'Buy' else 'red' if a == 'Sell' else 'gray' for a in actions],
                                         size=10),
                             name='Signal'))

    fig.update_layout(title='Actual vs Predicted Prices with Actions',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      template='plotly_dark')
    return fig

def plot_indicators(df):
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    fig = make_indicator_chart(df)
    return fig

def make_indicator_chart(df):
    fig = go.Figure()

    # MA
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], name='MA5'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA10'], name='MA10'))

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', yaxis='y2'))

    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', yaxis='y3'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='MACD Signal', yaxis='y3'))

    layout = go.Layout(
        title='Technical Indicators',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price'),
        yaxis2=dict(title='RSI', overlaying='y', side='right'),
        yaxis3=dict(title='MACD', anchor='free', overlaying='y', side='right', position=0.95),
        template='plotly_dark',
        height=600
    )

    fig.update_layout(layout)
    return fig

def generate_summary(actual, predicted):
    last_actual = actual.iloc[-1]
    last_pred = predicted.iloc[-1]
    diff_pct = (last_pred - last_actual) / last_actual * 100

    if diff_pct > 3:
        action = 'Buy'
    elif diff_pct < -3:
        action = 'Sell'
    else:
        action = 'Hold'

    return {
        'actual': round(last_actual, 2),
        'predicted': round(last_pred, 2),
        'difference': f"{diff_pct:.2f}%",
        'action': action
    }
