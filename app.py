import streamlit as st
import pandas as pd
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
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("tuned_xgboost_model.pkl")
    except:
        st.warning("⚠️ Model not loaded")
        return None

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
    data['is_weekend'].map({0:'Weekday',1:'Weekend'}).isin(day_type)
]

# -------------------------------
# TITLE
# -------------------------------
st.title("🚲 Bike Rental Analytics + Prediction")

# -------------------------------
# KPI METRICS
# -------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Total Rentals", int(filtered_data['cnt'].sum()))
c2.metric("Avg Rentals", round(filtered_data['cnt'].mean(), 2))
c3.metric("Peak Hour", int(filtered_data.groupby('hr')['cnt'].mean().idxmax()))

st.markdown("---")

# -------------------------------
# DONUT PIE CHART
# -------------------------------
st.subheader("🚦 Demand Distribution by Time Category")

pie_data = filtered_data.groupby('time_category')['cnt'].sum().reset_index()

fig_pie = px.pie(
    pie_data,
    values='cnt',
    names='time_category',
    hole=0.6,
    color='time_category',
    color_discrete_sequence=px.colors.qualitative.Bold
)

fig_pie.update_traces(textinfo='percent+label')

st.plotly_chart(fig_pie, use_container_width=True)

# -------------------------------
# HOURLY TREND
# -------------------------------
st.subheader("📈 Hourly Trends")

hourly = filtered_data.groupby('hr')['cnt'].mean().reset_index()
fig_line = px.line(hourly, x='hr', y='cnt', markers=True)
st.plotly_chart(fig_line, use_container_width=True)

# -------------------------------
# FEATURE TEMPLATE (MODEL MATCH)
# -------------------------------
MODEL_FEATURES = [
    'yr','holiday','workingday','temp','atemp','hum','windspeed',
    'day_of_week','is_weekend',
    'season_2','season_3','season_4',
    'weathersit_2','weathersit_3','weathersit_4',
    'mnth_2','mnth_3','mnth_4','mnth_5','mnth_6','mnth_7','mnth_8','mnth_9','mnth_10','mnth_11','mnth_12',
    'hr_1','hr_2','hr_3','hr_4','hr_5','hr_6','hr_7','hr_8','hr_9','hr_10','hr_11','hr_12',
    'hr_13','hr_14','hr_15','hr_16','hr_17','hr_18','hr_19','hr_20','hr_21','hr_22','hr_23',
    'weekday_1','weekday_2','weekday_3','weekday_4','weekday_5','weekday_6',
    'time_category_Low Demand','time_category_Morning Rush','time_category_Normal Hours'
]

def build_input(hour, temp, hum, windspeed, is_weekend):

    row = dict.fromkeys(MODEL_FEATURES, 0)

    # numeric
    row['temp'] = temp
    row['atemp'] = temp
    row['hum'] = hum
    row['windspeed'] = windspeed
    row['is_weekend'] = is_weekend

    # defaults
    row['yr'] = 1
    row['holiday'] = 0
    row['workingday'] = 0 if is_weekend else 1
    row['day_of_week'] = 6 if is_weekend else 2

    # hour
    if hour != 0:
        row[f'hr_{hour}'] = 1

    # weekday
    weekday = 6 if is_weekend else 2
    if weekday != 0:
        row[f'weekday_{weekday}'] = 1

    # defaults for missing UI inputs
    row['mnth_5'] = 1
    row['season_2'] = 1

    # time category
    if 7 <= hour <= 9:
        row['time_category_Morning Rush'] = 1
    elif 0 <= hour <= 5:
        row['time_category_Low Demand'] = 1
    else:
        row['time_category_Normal Hours'] = 1

    return pd.DataFrame([row])

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
day_type_input = p5.selectbox("Day Type", ["Weekday", "Weekend"])

is_weekend_val = 1 if day_type_input == "Weekend" else 0

if st.button("Predict Demand"):
    if model is not None:
        try:
            input_df = build_input(hour, temp, hum, windspeed, is_weekend_val)
            prediction = model.predict(input_df)[0]
            st.success(f"🚲 Predicted Rentals: {int(prediction)}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.warning("Model not available")

# -------------------------------
# FOOTER
# -------------------------------
st.info("⚠️ Model expects encoded features. For best results, use training pipeline.")
