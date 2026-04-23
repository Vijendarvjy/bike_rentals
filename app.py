import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title='🚲 Bike Rental Dashboard', layout='wide')

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def categorize_hour(hour):
    if 7 <= hour <= 9:
        return 'Morning Rush'
    elif 17 <= hour <= 19:
        return 'Evening Rush'
    elif 0 <= hour <= 5:
        return 'Low Demand'
    else:
        return 'Normal Hours'

def encode_time_category(cat):
    mapping = {
        'Low Demand': 0,
        'Normal Hours': 1,
        'Morning Rush': 2,
        'Evening Rush': 3
    }
    return mapping.get(cat, 1)

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('hour.csv')
    df['dteday'] = pd.to_datetime(df['dteday'])
    df['day_name'] = df['dteday'].dt.day_name()
    df['is_weekend'] = df['dteday'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
    df['time_category'] = df['hr'].apply(categorize_hour)
    return df

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("tuned_xgboost_model.pkl")
    except:
        st.warning("⚠️ Model not loaded")
        return None

# -------------------------------
# LOAD
# -------------------------------
data = load_data()
model = load_model()

# -------------------------------
# SIDEBAR FILTERS
# -------------------------------
st.sidebar.title("🔍 Filters")

day_type = st.sidebar.multiselect(
    "Day Type",
    ['Weekday', 'Weekend'],
    default=['Weekday', 'Weekend']
)

filtered_data = data[
    data['is_weekend'].map({0: 'Weekday', 1: 'Weekend'}).isin(day_type)
]

# -------------------------------
# TITLE
# -------------------------------
st.title("🚲 Bike Rental Analytics + Prediction")

# -------------------------------
# KPI METRICS
# -------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Rentals", int(filtered_data['cnt'].sum()))
col2.metric("Avg Rentals", round(filtered_data['cnt'].mean(), 2))
col3.metric("Peak Hour", int(filtered_data.groupby('hr')['cnt'].mean().idxmax()))

st.markdown("---")

# -------------------------------
# DONUT CHART
# -------------------------------
st.subheader("🚦 Demand Distribution by Time Category")

pie_data = (
    filtered_data
    .groupby('time_category')['cnt']
    .sum()
    .reset_index()
)

fig_pie = px.pie(
    pie_data,
    values='cnt',
    names='time_category',
    hole=0.6,
    color='time_category',
    color_discrete_sequence=px.colors.qualitative.Bold
)

st.plotly_chart(fig_pie, use_container_width=True)

# -------------------------------
# PREDICTION
# -------------------------------
st.header("🔮 Predict Bike Demand")

p1, p2, p3 = st.columns(3)

hour = p1.slider("Hour", 0, 23, 10)
temp = p2.slider("Temperature", 0.0, 1.0, 0.5)
hum = p3.slider("Humidity", 0.0, 1.0, 0.5)

p4, p5 = st.columns(2)
windspeed = p4.slider("Windspeed", 0.0, 1.0, 0.2)
is_weekend_input = p5.selectbox("Day Type", ["Weekday", "Weekend"])

# Encode features
time_category_raw = categorize_hour(hour)
time_category = encode_time_category(time_category_raw)
is_weekend_val = 1 if is_weekend_input == "Weekend" else 0

# Input dataframe (ALL NUMERIC)
input_df = pd.DataFrame({
    'hr': [hour],
    'temp': [temp],
    'hum': [hum],
    'windspeed': [windspeed],
    'is_weekend': [is_weekend_val],
    'time_category': [time_category]
})

if st.button("Predict Demand"):
    if model is not None:
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"🚲 Predicted Rentals: {int(prediction)}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.warning("Model not available")

# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------
st.header("📊 Feature Importance")

if model is not None and hasattr(model, "feature_importances_"):
    importance = pd.DataFrame({
        'Feature': model.feature_names_in_,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    fig_imp = px.bar(
        importance,
        x='Importance',
        y='Feature',
        orientation='h'
    )

    st.plotly_chart(fig_imp, use_container_width=True)

# -------------------------------
# HOURLY TREND
# -------------------------------
st.subheader("📈 Hourly Trends")

hourly = filtered_data.groupby('hr')['cnt'].mean().reset_index()

fig = px.line(hourly, x='hr', y='cnt', markers=True)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# FOOTER
# -------------------------------
st.info("Ensure model training used same feature encoding.")
