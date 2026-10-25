from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Restaurant Revenue Prediction")

st.title("Restaurant Revenue Prediction")
st.write("Predict restaurant revenue using Machine Learning.")

csv_file = Path(__file__).parent / "restaurant_data.csv"

if not csv_file.exists():
    st.error(f"CSV file nahi mili: {csv_file}")
    st.write("Check karo ki app.py aur restaurant_data.csv ek hi folder mein hain.")
    st.stop()

data = pd.read_csv(csv_file)

st.subheader("Restaurant Training Data")
st.dataframe(data)

features = ["customers", "average_order", "marketing"]
X = data[features]
y = data["revenue"]

model = LinearRegression()
model.fit(X, y)

st.subheader("Predict Revenue")

customers = st.number_input("Expected customers", min_value=1, value=200)
average_order = st.number_input("Average order value (₹)", min_value=1, value=300)
marketing = st.number_input("Marketing budget (₹)", min_value=0, value=5000)

if st.button("Predict Revenue"):
    prediction = model.predict(
        pd.DataFrame([{
            "customers": customers,
            "average_order": average_order,
            "marketing": marketing
        }])
    )[0]

    st.success(f"Predicted Revenue: ₹{prediction:,.2f}")