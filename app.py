import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Phone Addiction Predictor",
    page_icon="📱",
    layout="wide"
)

st.title("📱 Phone Addiction Level Predictor")

# Load model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("phone_addiction_dataset.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Remove target column
target = "Addiction_Level"

if target in df.columns:
    X = df.drop(columns=[target])
else:
    X = df.copy()

# Remove Name column if exists
if "Name" in X.columns:
    X = X.drop(columns=["Name"])

st.sidebar.header("Enter User Details")

input_data = {}

for col in X.columns:

    if X[col].dtype == "object":

        options = sorted(df[col].astype(str).unique())

        value = st.sidebar.selectbox(
            col,
            options
        )

        input_data[col] = value

    else:

        value = st.sidebar.number_input(
            col,
            value=float(X[col].mean())
        )

        input_data[col] = value

# Prediction
if st.sidebar.button("Predict"):

    input_df = pd.DataFrame([input_data])

    try:
        prediction = model.predict(input_df)[0]

        st.success(
            f"Predicted Addiction Level: {prediction:.2f}/10"
        )

        if prediction < 4:
            st.info("Low Addiction")
        elif prediction < 7:
            st.warning("Moderate Addiction")
        else:
            st.error("High Addiction")

    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.subheader("Dataset Statistics")
st.write(df.describe())