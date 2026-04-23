import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

# Set Page Config
st.set_page_config(page_title='Bike Rental Dashboard', layout='wide')

# 1. Load Data and Model
@st.cache_data
def load_data():
    df = pd.read_csv('hour.csv')
    df['dteday'] = pd.to_datetime(df['dteday'])
    df['day_name'] = df['dteday'].dt.day_name()
    df['is_weekend'] = df['dteday'].dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

    def categorize_hour(hour):
        if 7 <= hour <= 9: return 'Morning Rush'
        elif 17 <= hour <= 19: return 'Evening Rush'
        elif 0 <= hour <= 5: return 'Low Demand'
        else: return 'Normal Hours'

    df['time_category'] = df['hr'].apply(categorize_hour)
    return df

data = load_data()

# Sidebar
st.sidebar.title("Dashboard Filters")
selected_day_type = st.sidebar.multiselect("Select Day Type", options=['Weekday', 'Weekend'], default=['Weekday', 'Weekend'])
filtered_data = data[data['is_weekend'].isin(selected_day_type)]

st.title("🚲 Bike Rental Analytics Dashboard")
st.markdown("--- ")

# Layout Columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Weekday vs Weekend Distribution")
    fig_day_type = px.bar(
        filtered_data.groupby('is_weekend')['cnt'].mean().reset_index(),
        x='is_weekend', y='cnt', color='is_weekend', 
        color_discrete_sequence=px.colors.qualitative.Pastel,
        labels={'cnt': 'Avg Rentals', 'is_weekend': 'Day Type'}
    )
    st.plotly_chart(fig_day_type, use_container_width=True)

with col2:
    st.subheader("Peak vs Normal Hours Demand")
    fig_peak = px.pie(
        filtered_data.groupby('time_category')['cnt'].sum().reset_index(),
        values='cnt', names='time_category', 
        hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig_peak, use_container_width=True)

st.subheader("Hourly Rental Trends (Day-wise Comparison)")
hourly_data = filtered_data.groupby(['hr', 'is_weekend'])['cnt'].mean().reset_index()
fig_hourly = px.line(
    hourly_data, x='hr', y='cnt', color='is_weekend', 
    markers=True, line_shape='spline', color_discrete_sequence=['#636EFA', '#EF553B']
)
st.plotly_chart(fig_hourly, use_container_width=True)

st.subheader("Daily Usage Patterns")
daywise_avg = filtered_data.groupby('day_name')['cnt'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).reset_index()
fig_daywise = px.bar(
    daywise_avg, x='day_name', y='cnt', color='cnt', 
    color_continuous_scale='Viridis',
    labels={'cnt': 'Avg Total Rentals'}
)
st.plotly_chart(fig_daywise, use_container_width=True)

st.info("Note: To run this app locally, install streamlit and run: streamlit run app.py")
