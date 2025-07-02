import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the dataset
df = pd.read_csv('ntc.csv')

# App title
st.title("Stock Price Predictor (NTC)")

# Show the dataframe
st.subheader("Dataset Preview")
st.write(df.head())

# Plot High and Low prices
st.subheader("High and Low Price Over Time")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df['High'], label='High', color='red')
ax.plot(df['Low'], label='Low', color='blue')
ax.set_title('High and Low')
ax.set_xlabel('Index')
ax.set_ylabel('Price')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Train the model
X = df[['Open', 'High', 'Low']]
y = df['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Show metrics
st.subheader("Model Performance")
st.write(f"**Mean Absolute Error:** {mean_absolute_error(y_test, y_pred):.2f}")
st.write(f"**Mean Squared Error:** {mean_squared_error(y_test, y_pred):.2f}")
st.write(f"**R-squared:** {r2_score(y_test, y_pred):.2f}")

# User input for prediction
st.subheader("Predict Closing Price")
open_val = st.number_input("Open Price", min_value=0.0, value=950.0)
high_val = st.number_input("High Price", min_value=0.0, value=970.0)
low_val = st.number_input("Low Price", min_value=0.0, value=940.0)

if st.button("Predict"):
    prediction = model.predict([[open_val, high_val, low_val]])
    st.success(f"Predicted Closing Price: {prediction[0]:.2f}")
