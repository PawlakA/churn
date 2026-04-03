import streamlit as st
import pandas as pd
import mlflow.pyfunc
import os
cwd = os.getcwd()
print(cwd)

# Load model
model_file = os.path.join(cwd, 'src/serving/models/m-1ed936af9d1748ad9cef4e624c5951b2/artifacts')
model = mlflow.pyfunc.load_model(model_file)

# Page title with larger font
st.markdown("<h1 style='text-align: center; font-size: 48px;'>Churn Prediction Demo</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 20px;'>Enter feature values to predict churn.</p>", unsafe_allow_html=True)

# Two-column layout for sliders
col1, col2 = st.columns(2)

with col1:
    feature1 = st.slider("Average number of days between orders", min_value=0, max_value=500, value=0)
    feature2 = st.slider("Made multiple purchases? (0 or 1)", min_value=0, max_value=1, value=0)
    feature3 = st.slider("Recency", min_value=0, max_value=500, value=0)

with col2:
    feature4 = st.slider("Frequency", min_value=0, max_value=500, value=0)
    feature5 = st.slider("Monetary", min_value=0, max_value=5000, value=0)
    feature6 = st.slider("Number of returns / cancellations", min_value=0, max_value=100, value=0)

# Collect features into a pandas DataFrame
data = pd.DataFrame({
    "avg_days_between": [feature1],
    "has_multiple_purchases": [feature2],
    "Recency": [feature3],
    "Frequency": [feature4],
    "Monetary": [feature5],
    "returns": [feature6]
})

# Predict button
if st.button("Predict"):
    preds = model.predict(data)
    result = int(preds.tolist()[0])

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: blue; font-size: 32px;'>Prediction Result</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 28px;'>"
                f"{'⚠️ Likely to churn' if result == 1 else '✅ Not likely to churn'}</p>", unsafe_allow_html=True)
