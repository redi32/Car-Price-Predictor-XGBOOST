import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 2rem;
    }
    .price-value {
        font-size: 3rem;
        font-weight: bold;
        color: #2ecc71;
    }
    </style>
""", unsafe_allow_html=True)

# Load model, encoders, and categorical values
@st.cache_resource
def load_model_and_data():
    """Load the trained model, label encoders, and categorical values"""
    try:
        model = joblib.load('xgb_car_price_model.pkl')
        encoders = joblib.load('label_encoders.pkl')
        categorical_values = joblib.load('categorical_values.pkl')
        return model, encoders, categorical_values, None
    except FileNotFoundError as e:
        error_msg = f"Missing file: {str(e)}"
        return None, None, None, error_msg
    except Exception as e:
        error_msg = f"Error loading files: {str(e)}"
        return None, None, None, error_msg

# Load everything
model_obj, encoders, cat_values, error_msg = load_model_and_data()

# Default values in case files aren't loaded
DEFAULT_MAKES = ['Toyota', 'Ford', 'Honda', 'Chevrolet', 'Nissan']
DEFAULT_MODELS = ['Camry', 'Civic', 'Accord', 'Corolla', 'F-150']
DEFAULT_BODY = ['Sedan', 'SUV', 'Truck', 'Coupe', 'Van']
DEFAULT_TRANS = ['automatic', 'manual', 'cvt']

# Get the actual values or use defaults
if cat_values:
    MAKES = cat_values.get('makes', DEFAULT_MAKES)
    MODELS = cat_values.get('models', DEFAULT_MODELS)
    BODY_TYPES = cat_values.get('body_types', DEFAULT_BODY)
    TRANSMISSIONS = cat_values.get('transmissions', DEFAULT_TRANS)
else:
    MAKES = DEFAULT_MAKES
    MODELS = DEFAULT_MODELS
    BODY_TYPES = DEFAULT_BODY
    TRANSMISSIONS = DEFAULT_TRANS

# Title
st.markdown('<p class="main-header">🚗 Car Price Prediction System</p>', unsafe_allow_html=True)

