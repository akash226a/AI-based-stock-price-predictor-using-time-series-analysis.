import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set style and page config
sns.set(style='whitegrid')
st.set_page_config(page_title="📊 Stock Price Predictor", layout="centered")
st.title("📈 Predict Stock Close Price")
st.markdown("This app predicts the stock **Close Price** based on **Open Price** and **Volume**.")

# Load the data
df = pd.read_csv("AAPL.csv")

# Ensure required columns exist
required_cols = ['Date', 'Open', 'Volume', 'Close']
if all(col in df.columns for col in required_cols):
    # Convert Date to datetime for display and sorting
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Data summary and null check
    with st.expander("📄 Dataset Overview and Summary"):
        st.write("### Sample Data")
        st.dataframe(df[required_cols].head())
        
        st.write("### Data Summary")
        st.write(df[required_cols].describe())
        
        st.write("### Missing Values")
        st.write(df[required_cols].isnull().sum())

    # Correlation heatmap
    with st.expander("🔍 Feature Correlation Heatmap"):
        fig_corr, ax_corr = plt.subplots(figsize=(6,4))
        sns.heatmap(df[['Open', 'Volume', 'Close']].corr(), annot=True, cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)
    # Features and target
    X = df[['Open', 'Volume']]
    y = df['Close']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions for evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    with st.expander("📊 Model Evaluation"):
        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")

    # User input section
    st.subheader("🔢 Enter Stock Details to Predict Close Price")

    open_val = st.number_input("Enter Open Price", min_value=0.01, format="%.2f")
    volume_val = st.slider("Enter Volume", min_value=int(df['Volume'].min()), max_value=int(df['Volume'].max()), step=1000, value=int(df['Volume'].median()))

    if st.button("Predict Close Price"):
        user_input = pd.DataFrame([[open_val, volume_val]], columns=['Open', 'Volume'])
        prediction = model.predict(user_input)[0]
        st.success(f"📌 Predicted Close Price: **${prediction:.2f}**")

        # Suggestion logic based on recent close
        recent_close = df['Close'].iloc[-1]
        if prediction > recent_close * 1.01:
            st.markdown("📈 Suggestion: **BUY** - Expected rise.")
        elif prediction < recent_close * 0.99:
            st.markdown("📉 Suggestion: **SELL** - Expected drop.")
        else:
            st.markdown("⏸ Suggestion: **HOLD** - No significant change.")

    # Visualization: Actual vs Predicted scatter plot
    st.subheader("📉 Actual vs Predicted Close Prices (Test Set)")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_title("Actual vs Predicted Close Prices")
    ax.set_xlabel("Actual Close Price")
    ax.set_ylabel("Predicted Close Price")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # reference line
    st.pyplot(fig)

    # Visualization: Predicted vs Actual over time (test set)
    st.subheader("📈 Predicted vs Actual Close Price Over Time (Test Set)")
    test_dates = df.loc[y_test.index, 'Date']
    result_df = pd.DataFrame({
        'Date': test_dates,
        'Actual Close': y_test,
        'Predicted Close': y_pred
    }).sort_values('Date')

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(result_df['Date'], result_df['Actual Close'], label='Actual Close')
    ax2.plot(result_df['Date'], result_df['Predicted Close'], label='Predicted Close', linestyle='--')
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Close Price")
    ax2.set_title("Actual vs Predicted Close Price Over Time")
    ax2.legend()
    st.pyplot(fig2)

else:
    st.error("❌ Required columns (Date, Open, Volume, Close) not found in the dataset.")
