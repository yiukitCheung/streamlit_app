import pandas as pd
import numpy as np
import pymongo
import plotly.graph_objects as go
import streamlit as st
import plotly.subplots as sp

# MongoDB Configuration
DB_NAME = st.secrets['db_name']
PROCESSED_COLLECTION_NAME = st.secrets.processed_collection_name
ALERT_COLLECTION_NAME = st.secrets.alert_collection_name
def connect_to_mongo(db_name=DB_NAME):
    client = pymongo.MongoClient(**st.secrets["mongo"])
    return client[db_name]


def fetch_stock_data(collection, stock_symbol, interval):
    warehouse_interval = st.secrets.warehouse_interval
    if not warehouse_interval:
        raise ValueError("warehouse_interval is empty in st.secrets")
    return pd.DataFrame(list(collection.find({"symbol": stock_symbol,
                                            "interval": interval}, 
                                            {"_id": 0}))).sort_values(by=['date'])

def fetch_alert_data(collection, stock_symbol,interval):
    if not interval:
        raise ValueError("warehouse_interval is empty in st.secrets")
        # Fetch the data from MongoDB and convert to DataFrame
    data = list(collection.find({'symbol': stock_symbol, 'interval': interval}, {'_id': 0}))

    # Extract the alerts from the alert_dict
    for entry in data:
        if 'alerts' in entry and 'momentum_alert' in entry['alerts']:
            entry['momentum_alert'] = entry['alerts']['momentum_alert']['alert_type']
        if 'alerts' in entry and "velocity_alert" in entry['alerts']:
            entry['velocity_alert'] = entry['alerts']['velocity_alert']['alert_type']
        if 'alerts' in entry and '169ema_touched' in entry['alerts']:
            entry['touch_type'] = entry['alerts']['169ema_touched']['type']
        elif 'alerts' in entry and '13ema_touched' in entry['alerts']:
            entry['touch_type'] = entry['alerts']['13ema_touched']['type']
        else:
            entry['touch_type'] = np.nan

    # Convert the alert_dict to a DataFrame
    data = pd.DataFrame(data).sort_values(by=['date'])
    data = data.drop(columns=['alerts'])

    return data

def create_figure(filtered_df):

    row_height = [1]
    row = 1
    # Calculate the date range for the last 6 months
    end_date = filtered_df['date'].max() + pd.DateOffset(days=20)
    start_date = end_date - pd.DateOffset(months=6)
    
    # Calculate the price range for the last 6 months
    price_range = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]
    min_price = price_range['low'].min() * 0.95 
    max_price = price_range['high'].max() * 1.05
    
    fig = sp.make_subplots(rows=row, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=row_height)

    fig.add_trace(go.Candlestick(
        x=filtered_df['date'],
        open=filtered_df['open'],
        high=filtered_df['high'],
        low=filtered_df['low'],
        close=filtered_df['close'],
        name='price'), row=1, col=1)
    fig.update_xaxes(range=[start_date, end_date],title_text="Date", row=1, col=1)
    
    for ema in ['144ema', '169ema', '13ema', '8ema']:
        fig.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df[ema], mode="lines", name=ema), row=1, col=1)
        
    fig.update_xaxes(range=[start_date, end_date],title_text="Date", row=1, col=1)
    fig.update_yaxes(range=[min_price, max_price], title_text="Price", row=1, col=1)

    
    fig.update_layout(xaxis_rangeslider_visible=False, autosize=False, showlegend=False, height=1000, width=1200)

    return fig

def display_alerts(alert_df):
    # Display Alert
    today_alert = alert_df[alert_df['date'] == alert_df['date'].max()]

    # Function to map alert values to color and message
    def get_alert_color_and_message(alert_type, value):
        alert_mappings = {
            'velocity_alert': {
                'velocity_maintained': ('green', 'Velocity Maintained'),
                'velocity_weak': ('red', 'Velocity Weakened'),
                'velocity_loss': ('red', 'Velocity Loss'),
                'velocity_negotiating': ('red', 'Velocity Negotiating')
            },
            'touch_type': {
                'support': ('green', 'Support'),
               'resistance': ('red', 'Resistance')
            },
            'momentum_alert': {
                "accelerated": ('green', 'Accelerating'),
                'decelerated': ('red', 'Decelerating')
            }
        }
    
        # Default to grey color if value or alert type is not recognized
        color, message = alert_mappings.get(alert_type, {}).get(value, ('grey', 'Unknown Alert'))
        return color, message
    
    # Main function to display alerts
    def plot_alert(current_alert):
        # Loop through the relevant alert columns (e.g., velocity, candle)
        for column in ['velocity_alert', 'touch_type', 'momentum_alert']:
            alert_value = current_alert[column].values[0] if not current_alert[column].empty else None
            if pd.notna(alert_value):  # Only display if alert has a valid value
                alert_color, alert_message = get_alert_color_and_message(column, alert_value)
            
                # Display the alert with color-coded dot and message
                st.markdown(f"""
                    <div style="text-align: center;">
                    <span style="font-size:50px; color:{alert_color}">●</span>
                    <div style="font-size:16px; font-weight:bold; margin-top:10px; color:{alert_color};">{alert_message}</div>
                    </div>
                """, unsafe_allow_html=True)
                
    # Display the alerts            
    plot_alert(today_alert)
    
def static_analysis_page(processed_col, alert_col):
    # Set the page title and layout
    st.set_page_config(
        page_title="Long Term Alert Dashboard",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("<h1 style='text-align: center;'>Long Term Alert Dashboard</h1>", unsafe_allow_html=True)
    
    # Add a sidebar
    # Create a dropdown to select the stock
    stock_selector = st.sidebar.selectbox('Select Stock', options=sorted(processed_col.distinct("symbol")), index=0)
    # Create a dropdown to select the interval
    default_interval = '1D'
    interval_selector = st.sidebar.selectbox('Optimal Interval/ Select Interval',
                                             options=sorted(processed_col.distinct("interval")),
                                             index=sorted(processed_col.distinct("interval")). \
                                             index(default_interval) if default_interval in processed_col.distinct("interval")\
                                            else 0)

    alert_df = fetch_alert_data(alert_col, stock_selector, interval_selector)

    # Add an update button
    if st.sidebar.button("Update Data"):
        # fetch the latest data when the button is clicked
        processed_df = fetch_stock_data(processed_col, stock_selector, interval_selector)
        st.success("Data updated successfully!")
    else:
        # Display the existing data if the button is not clicked
        processed_df = fetch_stock_data(processed_col, stock_selector, interval_selector)
    # Create the figure
    fig = create_figure(processed_df)

    col1, col2 = st.columns([5, 1], vertical_alignment="top")
    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Display the alerts
        display_alerts(alert_df)
    
if __name__ == "__main__":
    # Connect to MongoDB and fetch the processed collection
    processed_collection = connect_to_mongo()[PROCESSED_COLLECTION_NAME]
    alert_collection = connect_to_mongo()[ALERT_COLLECTION_NAME]
    static_analysis_page(processed_collection, alert_collection)