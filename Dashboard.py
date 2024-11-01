import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from pymongo import MongoClient
import sys, os
import yfinance as yf
from datetime import datetime

# Ensure the correct path to the 'data' directory
from trading_strategy import DailyTradingStrategy, PickStock

# MongoDB Configuration
DB_NAME = st.secrets['db_name']
WAREHOUSE_INTERVAL = st.secrets.warehouse_interval
WAREHOUSE_INTERVAL_COLLECTION = '1d_data'
PROCESSED_COLLECTION = st.secrets.processed_collection_name
ALERT_COLLECTION = st.secrets.alert_collection_name

@st.cache_data
def get_most_current_trading_date() -> str:
    today = pd.to_datetime('today')
    current_date = today if today.hour > 14 else today - pd.Timedelta(days=1)
    # Check if today is a weekend or a public holiday
    if current_date.weekday() == 5:
        current_date -= pd.Timedelta(days=1)
    if current_date.weekday() == 6:
        current_date -= pd.Timedelta(days=2)
    if current_date.weekday() == 0 and current_date.hour <= 7:
        current_date -= pd.Timedelta(days=3)
    current_date = current_date.strftime('%Y-%m-%d')
    return current_date

@st.cache_resource
def initialize_mongo_client():
    client = MongoClient(**st.secrets["mongo"])
    return client

@st.cache_data
def fetch_index_return(symbol):
    if not WAREHOUSE_INTERVAL:
        raise ValueError("warehouse_interval is empty in st.secrets")
    start_of_year = datetime(datetime.now().year, 1, 1).strftime("%Y-%m-%d")

    # Fetch data from MongoDB for the specified symbol and date range
    Ticker = yf.Ticker(symbol)
    df = Ticker.history(
        start = start_of_year,
        interval=WAREHOUSE_INTERVAL).reset_index()

    # Ensure the DataFrame is sorted by date
    df = df.sort_values(by="Date")
    # Calculate cumulative returns based on the first available 'close' value
    df['cumulative_return'] = df['Close'].pct_change().fillna(0).add(1).cumprod()

    return df


def get_trade_data(data_collection, alert_collection):
    stock_candidates = PickStock(alert_collection).run()
    trades_history = DailyTradingStrategy(data_collection, alert_collection, stock_candidates)
    trades_history.execute_critical_trades()
    return trades_history.get_trades()

@st.cache_data
def compute_metrics(filtered_trades):
    # Verify required columns are present
    if 'profit/loss' not in filtered_trades.columns or 'total_asset' not in filtered_trades.columns:
        raise ValueError("The DataFrame must contain 'profit/loss' and 'total_asset' columns.")

    # Clean and convert profit/loss column to numeric decimal
    profit = filtered_trades['profit/loss'].str.replace('%', '').astype(float) / 100
    # Calculate the number of winning and losing trades
    win_trades = (profit > 0).sum()
    loss_trades = (profit <= 0).sum()

    # Calculate final trade profit rate
    final_trade_profit_rate = round((filtered_trades['total_asset'].iloc[-1] - 10000) / 100, 2)

    return [win_trades, loss_trades, final_trade_profit_rate]
    
st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")
st.sidebar.success("Select a page to view")

client =  MongoClient(**st.secrets["mongo"])
symbols = client[st.secrets['db_name']][st.secrets.processed_collection_name].distinct("symbol")

# Title
st.markdown("<h1 style='text-align: center;'>User Dashboard</h1>", unsafe_allow_html=True)

# ======================================================================================== #
# Simulated Trading Plot comparison and Simulated Trading Statistics Results Presentation  #
# ======================================================================================== #

# Plot the cumulative return of Index
line_chart = go.Figure()
for symbol in ['QQQ', 'SPY']:
    data = fetch_index_return(symbol)[['cumulative_return', 'Date']]

    line_chart.add_trace(go.Scatter(
        x=data['Date'],
        y=data['cumulative_return'],  # Ensure you use 'cumulative_return' as the y-axis
        mode='lines+markers',
        name=symbol,
        opacity=0.5
    ))

# Initialize MongoDB client and fetch the processed data
client = initialize_mongo_client()
symbols = client[st.secrets['db_name']][st.secrets['processed_collection_name']].distinct("symbol")

# Fetch trade data
df_trades = get_trade_data(client[DB_NAME][PROCESSED_COLLECTION],
                        client[DB_NAME][ALERT_COLLECTION])
# Compute metrics (e.g., win rate, loss rate, final profit rate)
win_trades, loss_trades, final_trade_profit_rate = compute_metrics(df_trades)

# Create columns in Streamlit for display (if using Streamlit)
col1, col2, col3 = st.columns(3, gap='medium')

# Ensure 'Exit_date' is in datetime format
df_trades['Exit_date'] = pd.to_datetime(df_trades['Exit_date'])
# Compute cumulative return of trades

# If there is only 1 trade
if len(df_trades) == 1:
    df_trades['Entry_date'] = pd.to_datetime(df_trades['Entry_date'])
    df_trades['Exit_date'] = pd.to_datetime(df_trades['Exit_date'])

    entry_date = df_trades.loc[0, 'Entry_date']
    exit_date = df_trades.loc[0, 'Exit_date']

    # Calculate cumulative return at exit
    entry_price = df_trades.loc[0, 'Entry_price']
    exit_price = df_trades.loc[0, 'Exit_price']
    cumulative_return_at_exit = 1 + (exit_price - entry_price) / entry_price

    # To visualize target data over a range, we need an artificial date range
    date_range = pd.date_range(start=entry_date, end=exit_date, freq='D')

    # Generate linearly interpolated cumulative return data
    cumulative_returns = [1 + (cumulative_return_at_exit - 1) * (i / (len(date_range) - 1)) for i in
                          range(len(date_range))]

    # Plotting the cumulative return of a single trade
    line_chart.add_trace(go.Scatter(
        x=date_range,
        y=cumulative_returns,
        mode='lines',
        name='Portfolio',
        marker_line_color="rgba(0,0,0,0.7)",
        opacity=1
    ))
    #
# If there are multiple trades
else:
    # Remove the '%' symbol and convert profit/loss to numeric
    df_trades['profit/loss'] = df_trades['profit/loss'].str.replace('%', '').astype(float) / 100

    # Calculate the cumulative return
    df_trades['cumulative_return'] = (1 + df_trades['profit/loss']).cumprod()

    # Add the cumulative return of the trades to the line chart
    line_chart.add_trace(go.Scatter(
        x=df_trades['Exit_date'],
        y=df_trades['cumulative_return'],
        mode='lines+markers',
        name='Portfolio',
        marker_line_color="rgba(0,0,0,0.7)",
        opacity=1
    ))
# Customize layout
line_chart.update_layout(
    title={
        "text": "Portfolio Return vs. SPY/QQQ Index",
        "x": 0.5,               # Center the title
        "xanchor": "center",     # Anchor the title at the center
        "yanchor": "top"         # Anchor the title at the top
    },
    xaxis_title="Date",
    yaxis_title="Cumulative Return",
    template="plotly_white",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    margin=dict(l=0, r=0, t=40, b=0),
    font=dict()
)
# CSS styling for metric containers
st.markdown("""
    <style>
    .metric-container {
        background-color: #f5f5f5;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        text-align: center;
        height: 100%;
    }
    .metric-label {
        font-weight: bold;
        font-size: 24px;
        margin: 0;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        margin-top: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ====================================================================== #
# Buy/ Hold/ Sell Suggested Stock Presentation based on Velocity Alerts  #
# ====================================================================== #

# Determine the color of the profit value based on its sign
profit_color = "green" if final_trade_profit_rate > 0 else "red"

with col1:
    st.markdown(f"""
        <div class="metric-container">
            <h3 class="metric-label">Final Trade Profit</h3>
            <p class="metric-value" style="color: {profit_color};">{final_trade_profit_rate}%</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-container">
            <h3 class="metric-label">Win Trades</h3>
            <p class="metric-value" style="color: green;">{win_trades}</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-container">
            <h3 class="metric-label">Loss Trades</h3>
            <p class="metric-value" style="color: red;">{loss_trades}</p>
        </div>
    """, unsafe_allow_html=True)

st.plotly_chart(line_chart)

most_recent_trade_date = pd.to_datetime(get_most_current_trading_date())
alert_collection = initialize_mongo_client()[DB_NAME][ALERT_COLLECTION]
current_alerts_dict = list(alert_collection.find({"date": {"$gte": most_recent_trade_date}}))

col1, col2 = st.columns(2)

def find_momentum_alert(data_dict: list, alert: str):
    results_set = {
        entry['symbol']
        for entry in data_dict
        if 'alerts' in entry and
           'momentum_alert' in entry['alerts'] and
           entry['alerts']['momentum_alert']['alert_type'] == alert
    }
    return list(results_set)

def find_velocity_alert(data_dict: list, alert: str):
    results_set = {
        entry['symbol']
        for entry in data_dict
        if 'alerts' in entry and
           'velocity_alert' in entry['alerts'] and
           entry['alerts']['velocity_alert']['alert_type'] == alert
    }
    return list(results_set)

with col1:
    results = find_momentum_alert(current_alerts_dict, 'accelerated')
    st.subheader("Buy Signal")
    if not results:
        st.write("No Opportunity found today, bored... 😴")
    else:
        # Add Buy specific styles
        st.markdown("""
            <style>
                .buy-container {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                }
                .buy-badge {
                    background-color: #4CAF50 !important;
                    color: white;
                    padding: 8px 12px;
                    border-radius: 5px;
                    font-size: 16px;
                    text-align: center;
                }
            </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="buy-container">', unsafe_allow_html=True)
        for symbol in results:
            st.markdown(f'<div class="buy-badge">{symbol}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    decelerated_results = find_momentum_alert(current_alerts_dict, 'decelerated')
    vel_loss_results = find_velocity_alert(current_alerts_dict, 'velocity_loss')
    results =  decelerated_results + vel_loss_results

    st.subheader("Sell Signal")
    if not results:
        st.write("No Disaster found today, Great...😊")
    else:
        # Add Sell specific styles
        st.markdown("""
            <style>
                .sell-container {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                }
                .sell-badge {
                    background-color: #FF0000 !important;
                    color: white;
                    padding: 8px 12px;
                    border-radius: 5px;
                    font-size: 16px;
                    text-align: center;
                }
            </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sell-container">', unsafe_allow_html=True)
        for symbol in results:
            st.markdown(f'<div class="sell-badge">{symbol}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)