import pandas as pd
import numpy as np
import pymongo
import plotly.graph_objects as go
import streamlit as st
import plotly.subplots as sp

# MongoDB Configuration
DB_NAME = st.secrets['mongo']['db_name']
PROCESSED_COLLECTION_NAME = st.secrets['mongo']['processed_collection_name']
ALERT_COLLECTION_NAME = st.secrets['mongo']['alert_collection_name']
def connect_to_mongo():
    client = pymongo.MongoClient(st.secrets["mongo"]["host"])
    return client[DB_NAME]


def fetch_stock_data(collection, stock_symbol, interval):
    warehouse_interval = st.secrets['mongo']['warehouse_interval']
    if not warehouse_interval:
        raise ValueError("warehouse_interval is empty in st.secrets")
    return pd.DataFrame(list(collection.find({"symbol": stock_symbol,
                                            "interval": interval,
                                            "instrument": "equity"}, 
                                            {"_id": 0}))).sort_values(by=['date'])

def fetch_alert_data(collection, stock_symbol, interval):
    if not interval:
        raise ValueError("warehouse_interval is empty in st.secrets")
        # Fetch the data from MongoDB and convert to DataFrame
    data = list(collection.find({'symbol': stock_symbol, 'interval': interval, 'instrument': 'equity'}, {'_id': 0}))

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

def candle_chart(filtered_df):

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

    fig.update_layout(xaxis_rangeslider_visible=False, autosize=False, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))

    return fig

def fundemental_chart(filtered_df):
    st.markdown("<h3 style='text-align: center;'>Fundamentals Analysis</h3>", unsafe_allow_html=True)

def display_alerts(alert_df):
    # Display Alert
    today_alert = alert_df[alert_df['date'] == alert_df['date'].max()]
    # Create columns for the three alert types
    col1, col2, col3 = st.columns(3)
    
    # Function to map alert values to color and message
    def get_alert_color_and_message(alert_type, value):
        alert_mappings = {
            'velocity_alert': {
                'velocity_maintained': ('green', 'Maintained'),
                'velocity_weak': ('orange', 'Weakened'), 
                'velocity_loss': ('red', 'Loss'),
                'velocity_negotiating': ('orange', 'Negotiating')
            },
            'touch_type': {
                'support': ('green', 'Support'),
                'resistance': ('red', 'Resistance'),
                
            },
            'momentum_alert': {
                "accelerated": ('green', 'Accelerating'),
                'decelerated': ('red', 'Decelerating'),
                
            }
        }
        
        # Default to grey color if value or alert type is not recognized
        color, message = alert_mappings.get(alert_type, {}).get(value, ('grey', 'No Alert'))
        return color, message

    # Display alerts in columns
    with col1:
        st.markdown("<h3 style='text-align: center;'>Up Trend Strength</h3>", unsafe_allow_html=True)
        if 'velocity_alert' in today_alert.columns and not today_alert['velocity_alert'].empty:
            alert_value = today_alert['velocity_alert'].values[0]
            if pd.notna(alert_value):
                color, message = get_alert_color_and_message('velocity_alert', alert_value)
                st.markdown(f"""
                    <div style="text-align: center;">
                        <span style="font-size:50px; color:{color}">●</span>
                        <div style="font-size:16px; font-weight:bold; margin-top:10px; color:{color};">{message}</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h3 style='text-align: center;'>Support/Resistance</h3>", unsafe_allow_html=True)
        if 'touch_type' in today_alert.columns:
            alert_value = today_alert['touch_type'].values[0]
            
            color, message = get_alert_color_and_message('touch_type', alert_value)

            st.markdown(f"""
                <div style="text-align: center;">
                    <span style="font-size:50px; color:{color}">●</span>
                    <div style="font-size:16px; font-weight:bold; margin-top:10px; color:{color};">{message}</div>
                </div>
                """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("<h3 style='text-align: center;'>Momentum</h3>", unsafe_allow_html=True)
        if 'momentum_alert' in today_alert.columns:
            alert_value = today_alert['momentum_alert'].values[0]
            color, message = get_alert_color_and_message('momentum_alert', alert_value)
            st.markdown(f"""
                <div style="text-align: center;">
                    <span style="font-size:50px; color:{color}">●</span>
                    <div style="font-size:16px; font-weight:bold; margin-top:10px; color:{color};">{message}</div>
                </div>
                """, unsafe_allow_html=True)

def static_analysis_page(processed_col, alert_col):
    # Set the page title and layout
    st.markdown("<h2 style='text-align: center;'>Long Term Alert Dashboard</h2>", unsafe_allow_html=True)
    chart_config_container = st.container()
    with chart_config_container:
        # Reduce space around selection components
        st.markdown("""
            <style>
            .css-1y0tads {padding-top: 0px; padding-bottom: 0px;}
            </style>
            """, unsafe_allow_html=True)
        
        symbol_col, interval_col = st.columns(2)
        with symbol_col:
            # Create a dropdown to select the stock
            stock_options = sorted(processed_col.find({'instrument':'equity'}).distinct("symbol"))
            stock_selector = st.selectbox('Select Stock', options=stock_options, index=0)
        with interval_col:
            # Create a dropdown to select the interval
            default_interval = '1D'
            interval_selector = st.selectbox('Optimal Interval/ Select Interval',
                                                options=sorted(processed_col.distinct("interval")),
                                                index=sorted(processed_col.distinct("interval")). \
                                                index(default_interval) if default_interval in processed_col.distinct("interval")\
                                                else 0)
    
    chart_container = st.container()
    with chart_container:
        alert_df = fetch_alert_data(alert_col, stock_selector, interval_selector)
        processed_df = fetch_stock_data(processed_col, stock_selector, interval_selector)
        
        candlesticks_chart, fundmentals_chart = st.columns([3, 1])
        with candlesticks_chart:
            # Create the figure
            fig = candle_chart(processed_df)
            st.plotly_chart(fig, use_container_width=True)

        with fundmentals_chart:
            fundemental_chart(processed_df)

    # Display the alerts
    display_alerts(alert_df)

def long_term_dashboard():
    # Connect to MongoDB and fetch the processed collection
    processed_collection = connect_to_mongo()[PROCESSED_COLLECTION_NAME]
    alert_collection = connect_to_mongo()[ALERT_COLLECTION_NAME]
    static_analysis_page(processed_collection, alert_collection)
