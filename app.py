import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title='🚲 Bike Rental Dashboard', layout='wide')

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("tuned_xgboost_model.pkl")

model = load_model()

# -------------------------------
# REQUIRED FEATURES (FROM MODEL)
# -------------------------------
MODEL_FEATURES = model.feature_names_in_

# -------------------------------
# HELPER: CREATE INPUT VECTOR
# -------------------------------
def create_feature_vector(input_dict):
    df = pd.DataFrame(columns=MODEL_FEATURES)
    df.loc[0] = 0  # initialize all as 0

    # Numerical features
    numeric_cols = ['temp', 'atemp', 'hum', 'windspeed', 'yr', 'holiday', 'workingday', 'day_of_week', 'is_weekend']
    for col in numeric_cols:
        if col in df.columns:
            df.at[0, col] = input_dict.get(col, 0)

    # One-hot encoding manually
    def set_one_hot(prefix, value):
        col_name = f"{prefix}_{value}"
        if col_name in df.columns:
            df.at[0, col_name] = 1

    # Apply encodings
    set_one_hot('hr', input_dict['hr'])
    set_one_hot('mnth', input_dict['mnth'])
    set_one_hot('weekday', input_dict['weekday'])
    set_one_hot('season', input_dict['season'])
    set_one_hot('weathersit', input_dict['weathersit'])

    # time_category encoding
    if input_dict['time_category'] == 'Low Demand':
        col = 'time_category_Low Demand'
    elif input_dict['time_category'] == 'Morning Rush':
        col = 'time_category_Morning Rush'
    else:
        col = 'time_category_Normal Hours'

    if col in df.columns:
        df.at[0, col] = 1

    return df

# -------------------------------
# TITLE
# -------------------------------
st.title("🚲 Bike Rental Prediction Dashboard")

# -------------------------------
# INPUTS
# -------------------------------
st.header("🔮 Predict Bike Demand")

c1, c2, c3 = st.columns(3)

hr = c1.slider("Hour", 0, 23, 10)
temp = c2.slider("Temperature", 0.0, 1.0, 0.5)
hum = c3.slider("Humidity", 0.0, 1.0, 0.5)

c4, c5, c6 = st.columns(3)

windspeed = c4.slider("Windspeed", 0.0, 1.0, 0.2)
mnth = c5.selectbox("Month", list(range(1, 13)))
weekday = c6.selectbox("Weekday (0=Sun)", list(range(7)))

c7, c8, c9 = st.columns(3)

season = c7.selectbox("Season (1-4)", [1,2,3,4])
weathersit = c8.selectbox("Weather (1-4)", [1,2,3,4])
is_weekend = c9.selectbox("Weekend?", [0,1])

# Derived features
workingday = 0 if is_weekend == 1 else 1
holiday = 0
yr = 1
day_of_week = weekday

# Time category
def categorize_hour(h):
    if 7 <= h <= 9:
        return 'Morning Rush'
    elif 17 <= h <= 19:
        return 'Morning Rush'  # merged to match model
    elif 0 <= h <= 5:
        return 'Low Demand'
    else:
        return 'Normal Hours'

time_category = categorize_hour(hr)

# Build input dict
input_dict = {
    'hr': hr,
    'temp': temp,
    'atemp': temp,
    'hum': hum,
    'windspeed': windspeed,
    'mnth': mnth,
    'weekday': weekday,
    'season': season,
    'weathersit': weathersit,
    'is_weekend': is_weekend,
    'workingday': workingday,
    'holiday': holiday,
    'yr': yr,
    'day_of_week': day_of_week,
    'time_category': time_category
}

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict Demand"):
    try:
        input_df = create_feature_vector(input_dict)
        prediction = model.predict(input_df)[0]
        st.success(f"🚲 Predicted Rentals: {int(prediction)}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------
st.header("📊 Feature Importance")

if hasattr(model, "feature_importances_"):
    importance = pd.DataFrame({
        'Feature': model.feature_names_in_,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(20)

    fig = px.bar(
        importance,
        x='Importance',
        y='Feature',
        orientation='h'
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# FOOTER
# -------------------------------
st.info("Model uses one-hot encoded features. Input is auto-transformed.")
