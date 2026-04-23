import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title='🚲 Bike Rental Dashboard', layout='wide')

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('hour.csv')
    df['dteday'] = pd.to_datetime(df['dteday'])
    df['day_name'] = df['dteday'].dt.day_name()
    df['is_weekend'] = df['dteday'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

    def categorize_hour(hour):
        if 7 <= hour <= 9:
            return 'Morning Rush'
        elif 17 <= hour <= 19:
            return 'Evening Rush'
        elif 0 <= hour <= 5:
            return 'Low Demand'
        else:
            return 'Normal Hours'

    df['time_category'] = df['hr'].apply(categorize_hour)
    return df
# -------------------------------
# ADVANCED PIE / DONUT CHART
# -------------------------------
st.subheader("🚦 Demand Distribution by Time Category")

pie_data = (
    filtered_data
    .groupby('time_category')['cnt']
    .sum()
    .reset_index()
    .sort_values(by='cnt', ascending=False)
)

# Highlight highest segment
pull_values = [0.1 if i == 0 else 0 for i in range(len(pie_data))]

fig_pie = px.pie(
    pie_data,
    values='cnt',
    names='time_category',
    hole=0.5,  # donut style
    color='cnt',
    color_continuous_scale='Turbo'  # 🔥 modern vibrant scale
)

fig_pie.update_traces(
    textinfo='percent+label+value',
    pull=pull_values,
    marker=dict(line=dict(color='#000000', width=1))
)

# Add center annotation (KPI style)
total = int(pie_data['cnt'].sum())

fig_pie.update_layout(
    annotations=[dict(
        text=f"Total<br><b>{total}</b>",
        x=0.5, y=0.5,
        font_size=18,
        showarrow=False
    )],
    showlegend=True
)

st.plotly_chart(fig_pie, use_container_width=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("tuned_xgboost_model.pkl")

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

filtered_data = data[data['is_weekend'].map({0:'Weekday',1:'Weekend'}).isin(day_type)]

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
col3.metric("Peak Hour", filtered_data.groupby('hr')['cnt'].mean().idxmax())

st.markdown("---")

# -------------------------------
# PREDICTION SECTION
# -------------------------------
st.header("🔮 Predict Bike Demand")

p1, p2, p3 = st.columns(3)

hour = p1.slider("Hour", 0, 23, 10)
temp = p2.slider("Temperature", 0.0, 1.0, 0.5)
hum = p3.slider("Humidity", 0.0, 1.0, 0.5)

p4, p5 = st.columns(2)
windspeed = p4.slider("Windspeed", 0.0, 1.0, 0.2)
is_weekend_input = p5.selectbox("Day Type", ["Weekday", "Weekend"])

# Feature Engineering for Prediction
def categorize_hour(hour):
    if 7 <= hour <= 9:
        return 'Morning Rush'
    elif 17 <= hour <= 19:
        return 'Evening Rush'
    elif 0 <= hour <= 5:
        return 'Low Demand'
    else:
        return 'Normal Hours'

time_category = categorize_hour(hour)
is_weekend_val = 1 if is_weekend_input == "Weekend" else 0

# Create input dataframe
input_df = pd.DataFrame({
    'hr': [hour],
    'temp': [temp],
    'hum': [hum],
    'windspeed': [windspeed],
    'is_weekend': [is_weekend_val],
    'time_category': [time_category]
})

# NOTE: If your model used encoding (OneHotEncoder), you MUST use pipeline
prediction = None

if st.button("Predict Demand"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🚲 Predicted Bike Rentals: {int(prediction)}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------
st.header("📊 Feature Importance")

if hasattr(model, "feature_importances_"):
    try:
        # If pipeline
        if hasattr(model, "named_steps"):
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
            importances = model.named_steps['model'].feature_importances_
        else:
            feature_names = model.feature_names_in_
            importances = model.feature_importances_

        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        fig_imp = px.bar(
            importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance"
        )

        st.plotly_chart(fig_imp, use_container_width=True)

    except Exception as e:
        st.error(f"Feature importance error: {e}")

# -------------------------------
# EXISTING VISUALS
# -------------------------------
st.subheader("Hourly Trends")

hourly = filtered_data.groupby(['hr'])['cnt'].mean().reset_index()

fig = px.line(hourly, x='hr', y='cnt', markers=True)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# FOOTER
# -------------------------------
st.info("Ensure model was trained with same feature pipeline.")
