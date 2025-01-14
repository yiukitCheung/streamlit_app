import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from pymongo import MongoClient
import yfinance as yf
import numpy as np
import redis, io, time
from datetime import datetime
from long_term import long_term_dashboard
from dependencies import search_stock, add_stock_to_database, check_symbol_yahoo, check_crypto_exists, add_crypto_to_database
# Ensure the correct path to the 'data' directory
from analyzer import ExpectedReturnRiskAnalyzer
from add_portfolio import existing_portfolio
from config.mongdb_config import load_mongo_config
from openai import OpenAI
import json
import streamlit_vertical_slider as svs
from redis import Redis
import gettext
import os
import json

# ============================
# ChatGPT Configuration
# ============================
ai_client = OpenAI(api_key=st.secrets['chatgpt']['api_key'])

# Define CSS for tooltips
tooltip_css = """
<style>
.centered-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100px; /* Adjust height as needed */
    }

    .tooltip {
    position: relative;
    display: inline-block;
    cursor: pointer;
    }

    .tooltip .tooltiptext {
    visibility: hidden;
    width: 200px;
    background-color: black;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 5px;
    position: absolute;
    z-index: 1;

    /* Center the tooltip horizontally */
    left: 50%;
    transform: translateX(-50%);

    /* Position tooltip above the element */
    bottom: 150%;
    }

    .tooltip:hover .tooltiptext {
    visibility: visible;
    }
    </style>
"""

# MongoDB Configuration
DB_NAME = st.secrets['mongo']['db_name']
WAREHOUSE_INTERVAL = st.secrets['mongo']['warehouse_interval']
WAREHOUSE_INTERVAL_COLLECTION = '1d_data'
LONG_TERM_ALERT_COLLECTION = st.secrets['mongo']['alert_collection_name']

# Redis Configuration
REDIS_HOST = st.secrets['redis']['host']
REDIS_PORT = st.secrets['redis']['port']
REDIS_PASSWORD = st.secrets['redis']['password']

# MongoDB Configuration
PROCESSED_COLLECTION = st.secrets['mongo']['processed_collection_name']
ALERT_COLLECTION = st.secrets['mongo']['alert_collection_name']
CANDIDATE_COLLECTION = st.secrets['mongo']['candidate_collection_name']
PORTFOLIO_COLLECTION = st.secrets['mongo']['portfolio_collection_name']
SANDBOX_COLLECTION = st.secrets['mongo']['alert_sandbox_name']


# AI Pre-configured Messages
PRECONFIGURED_MESSAGE = [
    {
        "role": "system",
        "content": "You are CondVest Advisor, a specialized financial AI assistant. You must strictly adhere to the following guidelines:\n\n"
                "1. Provide guidance ONLY based on the CondVest Platform's features and functionalities\n"
                "2. Focus on explaining the dashboard components and their practical usage\n"
                "3. Use clear, simple language accessible to beginners while maintaining accuracy\n"
                "4. Never provide general financial advice or recommendations outside the platform\n"
                "5. When explaining technical concepts:\n"
                "   - Break them down into simple terms\n"
                "   - Use concrete examples from the platform\n"
                "   - Relate them directly to platform features\n"
                "6. Respond only about:\n"
                "   - Dashboard features and metrics\n"
                "   - Platform tools and indicators\n"
                "   - How to interpret platform signals\n"
                "   - Platform-specific risk management\n\n"
                "Dashboard Overview:\n\n"
                "Market Overview (Top Left):\n"
                "- Shows overall market health through light indicators\n"
                "- Optimistic (Green): Strong uptrend\n"
                "- Neutral (Yellow): Slowing momentum\n"
                "- Pessimistic (Red): Weakening trend\n"
                "- Consolidating (Grey): Sideways movement\n\n"
                "Portfolio Tracking (Bottom Right):\n"
                "- Data-driven risk management and tracking service in real time\n"
                "- Greed/Risk indices\n"
                "- Real-time P/L tracking\n\n"
                "Profit/Loss Projection Tool (Middle Right):\n"
                "- Symbol-specific projections\n"
                "- Technical analysis metrics\n"
                "- Support/resistance levels\n"
                "- Trend indicators\n\n"
                "CondVest Pick (Top Right):\n"
                "- Buy signals based on trend reversals\n"
                "- Bounce opportunities at support\n"
                "- Built-in risk management\n"
                "- Stop-loss recommendations\n\n"
                "Stock Analysis Dashboard:\n"
                "- Up Trend Strength: Hold indicator\n"
                "- Support/Resistance: Entry/exit signals\n"
                "- Momentum: Day trading signals\n\n"
                "Comment on the market and the stock market:\n"
                "- All equity follows the trend, which is proven by the ema8, ema13, ema144, ema169\n"
                "- In a bullish trend, the ema8 and ema13 are above the ema144 and ema169\n"
                "- In a bearish trend, the ema8 and ema13 are below the ema144 and ema169\n"
                "- In a sideways trend, the ema8 and ema13 are around the ema144 and ema169\n"
                "- At the same time, all equity price reacts to the key level of the trend\n"
                "- The key level is the dense trading area, which may causes the price to bounce or break through depends on the trend\n"
                "- The key level is also the fibonacci retracement area, which may causes the price to bounce or break through depends on the trend\n"
                "- You should strictly follow the above rules to evaluate the inquired equity without mentioneing the tehcnical term, just proivde a alert for potential growth or risk\n"
                "Strictly avoid:\n"
                "- General market commentary\n"
                "- Non-platform trading advice\n"
                "- External tool recommendations\n"
                "- Personal opinions on trades\n\n"
                "Focus on helping users understand and effectively use CondVest's tools while maintaining strict adherence to platform-specific guidance."
    }
]

# ============================ #
# Data Section
# ============================ #

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
    try:
        client = MongoClient(st.secrets["mongo"]["host"])
    except Exception as e:
        st.error(f"Error initializing MongoDB client: {e}")
        return None
    return client

@st.cache_resource
def initialize_redis():
    """Initialize Redis connection with error handling"""
    try:
        redis_client = Redis(
            host=st.secrets["redis"]["host"],
            port=st.secrets["redis"]["port"],
            password=st.secrets["redis"]["password"],
            decode_responses=True  # This ensures responses are decoded to strings
        )
        # Test the connection
        redis_client.ping()
        return redis_client
    except Exception as e:
        st.error(f"Redis connection error: {e}")
        return None

@st.cache_data
def analyze_strategy_results():
    # Fetch the results from MongoDB (simulate this for now)
    results_df = pd.DataFrame(list(initialize_mongo_client()[DB_NAME][SANDBOX_COLLECTION].find({"instrument": 'equity'})))
    
    # Ensure the profit/loss column is used correctly and already in decimal format
    results_df['profit_loss_pct'] = results_df['profit/loss'].str.replace('%', '').astype(float) / 100  # Adjusted column name

    # Separate winning and losing trades based on profit_loss_pct
    profits = results_df.loc[results_df['profit_loss_pct'] > 0, 'profit_loss_pct']
    losses = results_df.loc[results_df['profit_loss_pct'] <= 0, 'profit_loss_pct']

    # Calculate win rate
    total_trades = len(results_df)
    win_trades = len(profits)
    win_rate = win_trades / total_trades if total_trades > 0 else 0

    # Calculate average profit and average loss
    avg_profit = profits.mean() if not profits.empty else 0
    avg_loss = losses.mean() if not losses.empty else 0

    # Return results in a dictionary
    return {
        "win_rate": round(win_rate * 100, 2),  # Convert to percentage
        "avg_profit": round(avg_profit * 100, 2),  # Convert to percentage
        "avg_loss": round(avg_loss * 100, 2),  # Convert to percentage
        "best_pick": round(profits.max() * 100, 2),  # Convert to percentage
        "worst_pick": round(losses.min() * 100, 2),  # Convert to percentage
        "total_trades": total_trades
    }

@st.cache_data
def fetch_data(instrument, interval):
    warehouse_interval = st.secrets['mongo']['warehouse_interval']
    if not warehouse_interval:
        raise ValueError("warehouse_interval is empty in st.secrets")

    redis_client = initialize_redis()
    collection_obj = initialize_mongo_client()[DB_NAME][PROCESSED_COLLECTION]
    redis_key = f"instrument_data:{instrument}:{interval}"
    try:
        # Check if data is cached in Redis and deserialize in one step
        if cached_data := redis_client.get(redis_key):
            data = pd.read_json(cached_data)
            cached_date = data['date'].max()
            
            if cached_date.date() < pd.to_datetime('today').date():
                
                # Use MongoDB projection and sorting at database level
                cursor = collection_obj.find(
                    {
                        "instrument": instrument,
                        "interval": interval,
                        "date": {"$gte": pd.Timestamp.now() - pd.Timedelta(days=1825)}
                    },
                    {"_id": 0}
                ).sort("date", 1)
        
        else:
            # Use MongoDB projection and sorting at database level
            cursor = collection_obj.find(
                {
                    "instrument": instrument,
                    "interval": interval,
                    "date": {"$gte": pd.Timestamp.now() - pd.Timedelta(days=1825)}
                },
                {"_id": 0}
            ).sort("date", 1)
            
            # Create DataFrame directly from cursor
            data = pd.DataFrame(cursor)
            # Cache the result with compression
            redis_client.setex(
                redis_key,
                120,
                data.to_json(orient="records", date_format='iso')
            )
            
        return data

    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error

def fetch_alert_data(instrument, symbol):
    try:
        collection_obj = initialize_mongo_client()[DB_NAME][ALERT_COLLECTION]
        query = {"instrument": instrument, "symbol": symbol}
        cursor = collection_obj.find(query)
        return pd.DataFrame(list(cursor))
    except Exception as e:
        st.error(f"System Error, Please wait...: {str(e)}")
        return pd.DataFrame()

def fetch_return_data(instrument):
    if not WAREHOUSE_INTERVAL:
        raise ValueError("warehouse_interval is empty in st.secrets")

    # Fetch data from MongoDB for the specified symbol and date range
    df = fetch_data(instrument, 1)
    
    # Calculate cumulative returns based on the first available 'close' value
    distinct_symbols = df['symbol'].unique()
    symbol_dfs = []
    for symbol in distinct_symbols:
        symbol_df = df.loc[df['symbol'] == symbol]
        symbol_df.loc[:, 'cumulative_return'] = symbol_df.loc[:, 'close'].pct_change().fillna(0).add(1).cumprod()
        symbol_dfs.append(symbol_df)
    df = pd.concat(symbol_dfs)
    
    return df

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

def find_alert_symbols(data_dict: list, alert_type: str):
    results_set = set()
    for entry in data_dict:
        if alert_type in entry and entry[alert_type]:
            results_set.update(entry[alert_type])  # Add all symbols in the array to the set
    return list(results_set)

@st.cache_data
def find_velocity_alert(data_dict: list, alert: int):
    results_set = {
        entry['symbol']
        for entry in data_dict
        if 'alerts' in entry and
        'main_accumulating' in entry['alerts'] and
        entry['main_accumulating'] == alert
    }
    return list(results_set)

def portfolio_chart(username: str):
    
    def get_alert_color(alert_data):
        """Determine the color of the dot based on the velocity alert type."""
        if "velocity_loss" in alert_data['alert_type']:
            return "red", "Pessimistic"  # Velocity loss
        elif "velocity_weak" in alert_data['alert_type']:
            return "#FFE082", "Flag for Risk"  # Velocity weak - using warmer warning yellow
        elif "velocity_negotiating" in alert_data['alert_type']:
            return "#FFE082", "Flag for Risk"  # Velocity negotiating - using warmer warning yellow
        elif "velocity_maintained" in alert_data['alert_type']:
            return "green", "Optimistic"  # Velocity maintained
        
        return "grey", "Neutral"  # Default if no relevant alert is found

    collection_obj = initialize_mongo_client()[DB_NAME][PORTFOLIO_COLLECTION]
    alert_collection = initialize_mongo_client()[DB_NAME][ALERT_COLLECTION]
    
    # Fetch the user's portfolio
    portfolio = collection_obj.find_one({'username': username}, projection={'portfolio': True, '_id': False})
    
    # Check if portfolio exists and is not empty
    if not portfolio or 'portfolio' not in portfolio or not portfolio['portfolio']:
        return False
        
    # Portfolio exists and has stocks
    portfolio = portfolio['portfolio']     
    
    # Create a container for the header and portfolio items
    with st.container():
        # Header row with interval labels
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: center; max-width: 600px; margin: 20px auto;">
                <div style="width: 100px; text-align: left;">
                    <span style="font-size: 18px; font-family: 'Quicksand', 'Varela Round', sans-serif; color: #2c3e50; font-weight: 400;">Symbol</span>
                </div>
                <div style="width: 250px; display: flex; justify-content: space-between; margin-right: 20px;">
                    <span style="font-size: 18px; font-family: 'Quicksand', 'Varela Round', sans-serif; color: #2c3e50; font-weight: 400;">Short</span>
                    <span style="font-size: 18px; font-family: 'Quicksand', 'Varela Round', sans-serif; color: #2c3e50; font-weight: 400;">Mid</span>
                    <span style="font-size: 18px; font-family: 'Quicksand', 'Varela Round', sans-serif; color: #2c3e50; font-weight: 400;">Long</span>
                </div>
            </div>
            """, unsafe_allow_html=True
        )

            
        # Display each stock in the portfolio and its alert status
        for symbol, _ in portfolio.items():
            symbol = symbol.upper()
            alerts = alert_collection.find(
                {
                    'symbol': symbol,
                    'interval': {'$in': [1, 3, 5]}
                },
                projection={'alerts': True, '_id': False, 'interval': True}
            ).sort('date', -1)

            # Create a dictionary to store the latest alert for each interval
            alert_dict = {}
            for alert in alerts:
                interval = alert['interval']
                if interval not in alert_dict:  # Only keep first (latest) alert per interval
                    alert_dict[interval] = alert

            short_alert_data = alert_dict.get(1, {})
            mid_alert_data = alert_dict.get(3, {})
            long_alert_data = alert_dict.get(5, {})
            intervals = [
                (short_alert_data, 'Short'),
                (mid_alert_data, 'Mid'), 
                (long_alert_data, 'Long')
            ]
            
            dots_html = ""
            for alert_data, interval_name in intervals:
                if alert_data and 'velocity_alert' in alert_data['alerts']:
                    velocity_alert = alert_data['alerts'].get('velocity_alert', {})
                    dot_color, _ = get_alert_color(velocity_alert)
                else:
                    dot_color = "white"
                dots_html += f'<div style="width: 20px; height: 20px; background-color: {dot_color}; border-radius: 75%; border: 0px solid #ccc;" title="{interval_name}"></div>'
                
            st.markdown(f"""
                <div style="display: flex; align-items: center; justify-content: center; max-width: 600px; margin: 25px auto;">
                    <div style="width: 100px;">
                        <span style="font-size: 16px; font-weight: bold; color: #2c3e50;">{symbol}</span>
                    </div>
                    <div style="width: 250px; display: flex; justify-content: space-between; margin-right: 20px;">
                        {dots_html}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Legend with adjusted spacing
        st.markdown("""
            <div style="display: flex; justify-content: center; align-items: center; margin-top: 30px; margin-bottom: 20px; max-width: 600px; margin-left: auto; margin-right: auto;">
                <div style="display: flex; align-items: center; margin-right: 20px;">
                    <div style="width: 10px; height: 10px; background-color: green; border-radius: 50%; margin-right: 5px;"></div>
                    <span style="font-size: 12px;">Optimistic</span>
                </div>
                <div style="display: flex; align-items: center; margin-right: 20px;">
                    <div style="width: 10px; height: 10px; background-color: #FFE082; border-radius: 50%; margin-right: 5px;"></div>
                    <span style="font-size: 12px;">Flag for Risk</span>
                </div>
                <div style="display: flex; align-items: center; margin-right: 20px;">
                    <div style="width: 10px; height: 10px; background-color: red; border-radius: 50%; margin-right: 5px;"></div>
                    <span style="font-size: 12px;">Pessimistic</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 10px; height: 10px; background-color: grey; border-radius: 50%; margin-right: 5px;"></div>
                    <span style="font-size: 12px;">Neutral</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    return True

def overview_chart(instrument: str, selected_symbols: str, chart_type: str, selected_interval: int):
    # Plot the cumulative return of Index
    if chart_type == "Cumulative Return":
        cum_return_chart = go.Figure()
        
        # Get the data and filter for selected symbol in one step
        cum_return_data = fetch_return_data(instrument)
        data = cum_return_data.loc[cum_return_data['symbol'] == selected_symbols, ['cumulative_return', 'date', 'symbol','interval']]
        data = data.sort_values(by='date', ascending=True)
        # Add the trace to the chart using the filtered data directly
        cum_return_chart.add_trace(go.Scatter(
            x=data.loc[:, 'date'],
            y=data.loc[:, 'cumulative_return'], 
            mode='lines+markers',
            name=selected_symbols,
            opacity=0.5
        ))
        # Update the layout of the chart
        cum_return_chart.update_layout(
            title={
                "text": f"{selected_symbols} YTD Return",
                "x": 0.5,  # Center the title
                "xanchor": "center",  # Anchor the title at the center
                "yanchor": "top"  # Anchor the title at the top
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
            ))
        chart = cum_return_chart
        
    elif chart_type == "Candlesticks":
        data = fetch_data(instrument, 1).loc[:, ['date', 'open', 'high', 'low', 'close', 'symbol']]
        
        # Filter the data for the selected symbol
        distinct_symbols = data['symbol'].unique() if not selected_symbols else selected_symbols
        data = data.loc[data['symbol'] == distinct_symbols]
        
        if selected_interval == 1:
            filtered_df = data.loc[data['date'] >= pd.to_datetime(data['date'].max()) - pd.Timedelta(days=365)]
        elif selected_interval == 3:
            filtered_df = data.loc[data['date'] >= pd.to_datetime(data['date'].max()) - pd.Timedelta(days=730)]
        elif selected_interval == 5:
            filtered_df = data.loc[data['date'] >= pd.to_datetime(data['date'].max()) - pd.Timedelta(days=1825)]
        elif selected_interval == 8:
            filtered_df = data.loc[data['date'] >= pd.to_datetime(data['date'].max()) - pd.Timedelta(days=3650)]
        else:
            filtered_df = data
        
        # Create the candlestick chart
        candlestick_chart = go.Figure()
        candlestick_chart.add_trace(go.Candlestick(
            x=filtered_df.loc[:,'date'].astype(str),
            open=filtered_df.loc[:,'open'],
            high=filtered_df.loc[:,'high'], 
            low=filtered_df.loc[:,'low'],
            close=filtered_df.loc[:,'close'],
            name='price'
        ))

        # Update y-axis properties with padding
        candlestick_chart.update_yaxes(
            range=[min(filtered_df.loc[:,'close']) * 0.9, max(filtered_df.loc[:,'close']) * 1.1], 
            title='Value',
            showgrid=True,  
            zeroline=True 
        )
        
        # Update layout to include range selector but without the range slider
        
        candlestick_chart.update_layout(
            title={
                "text": f"{selected_symbols} Candlesticks", 
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top"
            },
            
            xaxis_rangeslider_visible=False,
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom", 
                y=1.02,
                xanchor="right",
                x=1
            ),
        )

        candlestick_chart.update_yaxes(
            autorange = True,
            fixedrange = False
        )
        chart = candlestick_chart
        
    elif chart_type == 'Line Chart':
        line_chart = go.Figure()
        
        # Get the data and filter for selected symbol in one step
        data = fetch_data(instrument, 1).loc[:, ['close', 'date', 'symbol']]
        data = data.loc[data['symbol'] == selected_symbols]
        data = data.sort_values(by='date', ascending=True)
        
        # Filter the data based on the selected interval
        if selected_interval == 1:
            filtered_df = data.loc[data['date'] >= pd.to_datetime(data['date'].max()) - pd.Timedelta(days=365)]
        elif selected_interval == 3:
            filtered_df = data.loc[data['date'] >= pd.to_datetime(data['date'].max()) - pd.Timedelta(days=730)]
        elif selected_interval == 5:
            filtered_df = data.loc[data['date'] >= pd.to_datetime(data['date'].max()) - pd.Timedelta(days=1825)]
        elif selected_interval == 8:
            filtered_df = data.loc[data['date'] >= pd.to_datetime(data['date'].max()) - pd.Timedelta(days=3650)]
        else:
            filtered_df = data
        
        line_chart.add_trace(go.Scatter(
            x=filtered_df['date'],
            y=filtered_df['close'],
            fill='tozeroy',
            name=selected_symbols,
            opacity=0.5
        ))
        
        # Update y-axis properties with padding
        line_chart.update_yaxes(
            range=[min(filtered_df['close']) * 0.9, max(filtered_df['close']) * 1.1],  # Adjust the range to add padding
            title='Value',
            showgrid=True,  # Optional: show gridlines for better readability
            zeroline=True  # Optional: show a zero line if needed
        )
        # Update layout to include range selector and range slider
        line_chart.update_layout(
            xaxis_rangeslider_visible=False,
            title={
                "text": f"{selected_symbols} YTD Line Chart",
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top"
            },
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        chart = line_chart
    
    return chart

def calculate_expected_value(symbol: str, instrument: str, selected_period: int):
    """Separate function for the actual calculation logic"""
    exp_return_risk_analyzer = ExpectedReturnRiskAnalyzer()
    expected_support, expected_resistance = exp_return_risk_analyzer.find_sup_res(symbol.upper(), selected_period)
    
    if not expected_support or not expected_resistance:
        return 0, 0, 0
    
    # Find the ema support and resistance
    for entry in expected_support:
        if entry == 'emas' and expected_support[entry] != float('-inf'):
            support = expected_support[entry]
            break
        elif entry == 'fibonacci' and expected_support[entry] != float('-inf'):
            support = expected_support[entry]
            break
        elif entry == 'dense_area' and expected_support[entry] != float('-inf'):
            support = expected_support[entry]
            break
    
    for entry in expected_resistance:
        if entry == 'emas' and expected_resistance[entry] != float('inf'):
            resistance = expected_resistance[entry]
            break
        elif entry == 'fibonacci' and expected_resistance[entry] != float('inf'):
            resistance = expected_resistance[entry]
            break
        elif entry == 'dense_area' and expected_resistance[entry] != float('inf'):
            resistance = expected_resistance[entry]
            break
    
    cur_price = fetch_data(instrument, 1).loc[fetch_data(instrument, 1)['symbol'] == symbol.upper()].iloc[-1]['close']
    expected_loss = float(f"{((support - cur_price) / cur_price) * 100:.2f}")
    expected_profit = float(f"{((resistance - cur_price) / cur_price) * 100:.2f}")
    profit_loss_ratio = abs(float(f"{expected_profit / expected_loss:.2f}"))
    
    return expected_loss, expected_profit, profit_loss_ratio

@st.cache_data
def find_expected_value(symbol: str, instrument:str, selected_period: int):
    # Initialize Redis connection
    redis_client = initialize_redis()
    if not redis_client:
        return calculate_expected_value(symbol, instrument, selected_period)
    
    try:
        symbol = symbol.upper()
        cache_key = f"expected_value:{symbol}:{selected_period}"  # Changed key format
        live_cached_key = f"live_trade:{symbol}"
        
        # Clear existing keys if they're of wrong type
        try:
            redis_client.delete(cache_key)
            redis_client.delete(live_cached_key)
        except:
            pass
            
        # Try to get cached results
        try:
            cached_live_result = redis_client.get(live_cached_key)
            if cached_live_result:


                values = cached_live_result.split(',')
                if len(values) == 3:
                    expected_loss, expected_profit, profit_loss_ratio = map(float, values)
                    return expected_loss, expected_profit, profit_loss_ratio
        except:
            pass
            
        try:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                values = cached_result.split(',')
                if len(values) == 3:
                    expected_loss, expected_profit, profit_loss_ratio = map(float, values)
                    return expected_loss, expected_profit, profit_loss_ratio
        except:
            pass

        # Calculate new values
        expected_loss, expected_profit, profit_loss_ratio = calculate_expected_value(symbol, instrument, selected_period)
        
        # Cache the new results as a string
        try:
            cache_value = f"{expected_loss},{expected_profit},{profit_loss_ratio}"
            redis_client.setex(cache_key, 3600, cache_value)  # Cache for 1 hour
        except:
            pass
        
        return expected_loss, expected_profit, profit_loss_ratio
            
    except Exception as e:
        return calculate_expected_value(symbol, instrument, selected_period)

@st.cache_data
def compute_portfolio_metrics(username: str):
    
    # Initialize MongoDB connection
    collection_obj = initialize_mongo_client()[DB_NAME][PORTFOLIO_COLLECTION]
    
    # Initialize Redis connection
    redis_client = initialize_redis()
    cache_key = f"portfolio_metrics_{username}"

    # Try to get cached results
    cached_result = redis_client.get(cache_key)
    if cached_result:
        
        metrics = json.loads(cached_result)
        return metrics

    # Initialize metrics if not exist
    if not collection_obj.find_one({'username': username}, projection={'metrics': True, '_id': False}):
        collection_obj.update_one({'username': username}, {'$set': {'metrics': {}}})

    # Get user portfolio
    user_portfolio = collection_obj.find_one({'username': username}, projection={'portfolio': True, '_id': False})
    if not user_portfolio or 'portfolio' not in user_portfolio or not user_portfolio['portfolio']:
        return False
        
    portfolio_df = pd.DataFrame(user_portfolio['portfolio']).T
    # Get current prices
    portfolio_df['shares'] = pd.to_numeric(portfolio_df['shares'])
    portfolio_df['avg_price'] = pd.to_numeric(portfolio_df['avg_price'])
    portfolio_df['total_cost'] = portfolio_df['shares'] * portfolio_df['avg_price']
    
    total_possible_downside = 0
    
    # Process each instrument separately
    for instrument in portfolio_df['instrument'].unique():
        instrument_df = portfolio_df[portfolio_df['instrument'] == instrument]
        symbol_list = instrument_df.index.tolist()
        
        # Get current prices for this instrument's symbols
        current_prices = fetch_data(instrument, 1)
        current_prices = current_prices[current_prices['symbol'].isin(symbol_list)]
        current_prices = current_prices.groupby('symbol')['close'].last()
        
        # Update only the rows for this instrument
        portfolio_df.loc[symbol_list, 'current_close'] = portfolio_df.loc[symbol_list].index.map(current_prices)
        
        # Calculate risk index
        
        for symbol in symbol_list:
            instrument = portfolio_df.loc[symbol, 'instrument']
            possible_downside = find_expected_value(symbol, instrument, 1)[0]
            portfolio_df.loc[symbol, 'possible_downside'] = possible_downside
            
            total_possible_downside += portfolio_df.loc[symbol, 'shares'] * possible_downside
            
    risk_index = total_possible_downside / portfolio_df['total_cost'].sum()
    
    # Convert current_close to numeric after all updates
    portfolio_df['current_close'] = pd.to_numeric(portfolio_df['current_close'])
    
    # Calculate metrics
    portfolio_df['current_value'] = portfolio_df['shares'] * portfolio_df['current_close'] 
    portfolio_df['return'] = (portfolio_df['current_value'] - portfolio_df['total_cost']) / portfolio_df['total_cost']
    portfolio_df['return'] = portfolio_df['return'].apply(lambda x: round(x, 3))
    
    overall_return = portfolio_df['return'].mean()
    # Calculate greed index
    greed_index = min(100, round(50 * np.log(1 + 2 * abs(overall_return)) / np.log(3), 2))
    
    # Prepare metrics dict
    metrics = {
        'date': pd.to_datetime('today').strftime('%Y-%m-%d'),
        'overall_return': overall_return,
        'risk_index': risk_index,
        'greed_index': greed_index,
    }

    # Update MongoDB
    collection_obj.update_one(
        {'username': username}, 
        {'$set': {'metrics': metrics}}
    )

    # Cache the results for 1 hour (3600 seconds)
    redis_client.setex(cache_key, 3600, json.dumps(metrics))
    
    return metrics

def increment_dashboard_view_count():
    """Increment the dashboard view count in Redis."""
    try:
        redis_client = initialize_redis()
        redis_client.incr("dashboard_view_count")
    except Exception as e:
        st.error(f"Error incrementing view count: {e}")

def get_dashboard_view_count():
    """Retrieve the dashboard view count from Redis."""
    try:
        redis_client = initialize_redis()
        return redis_client.get("dashboard_view_count") or 0
    except Exception as e:
        st.error(f"Error retrieving view count: {e}")
        return 0

# ============================ #
# Visualization Section        #
# ============================ #

def symbol_exp_return_section(main_col):
    st.markdown("""
            <div style="text-align: center; font-size: 24px; font-weight: bold; color: #2c3e50;">
                {}
            </div>
    """.format(_("Check How Much You Can Earn in a Week 🔍")), unsafe_allow_html=True) 
    
    stock_col, crypto_col = main_col.columns([1, 1])
    with stock_col:
        search_stock_symbol = st.text_input(_("Search Stock"), key="search_stock", help=_("A decent stock should have Profit/Loss ratio of 1 or more"))
        
        if search_stock_symbol:
            if search_stock(search_stock_symbol) != None:
                with st.container():
                    toggle_slider = st.slider(_("💰 Investment Amount ($)"), min_value=100, max_value=10000, value=100, step=1000, key=f"toggle_{search_stock_symbol}")
                    # Display metrics for each period in columns
                    expected_loss, expected_profit, profit_loss_ratio = find_expected_value(search_stock_symbol, 'equity', 3)
                    # Expected Earnings        
                    expected_earnings = expected_profit / 100 * toggle_slider

                    if profit_loss_ratio >= 1:
                        expected_loss = expected_loss / 100 * toggle_slider
                        st.markdown(f"""
                            <div style='display: flex; justify-content: space-around;'>
                                <div style='color: #4CAF50; text-align: center;'>
                                    <p style='font-size:24px; font-weight:bold'> {_('Expected Earnings')}: ${expected_earnings:+,.2f}</p>
                                </div>
                                <div style='color: #4CAF50; text-align: center;'>
                                    <p style='font-size:24px; font-weight:bold'>{_('BUY')} 📈</p>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        expected_loss = expected_loss / 100 * toggle_slider
                        st.markdown(f"""
                            <div style='display: flex; justify-content: space-around;'>
                                <div style='color: #FF0000; text-align: center;'>
                                    <p style='font-size:24px; font-weight:bold'>{_('Expected Loss')}: ${expected_loss:+,.2f}</p>
                                </div>
                                <div style='color: #FF0000; text-align: center;'>
                                    <p style='font-size:24px; font-weight:bold'>{_('SELL')} 📉</p>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

            else:
                st.error(f"{_('Stock')} {search_stock_symbol} {_('not found, do you want to contribute this stock to our database?')}")
                if st.button(_("✨ Contribute New Stock"), use_container_width=True):
                    if check_symbol_yahoo(search_stock_symbol.upper()):
                        search_stock_symbol = search_stock_symbol.upper()
                        full_name = yf.Ticker(search_stock_symbol).info.get('longName')
                        if add_stock_to_database(search_stock_symbol, full_name):
                            st.success(_("🎉 Thanks for your contribution! Your stock will be added to the database tomorrow."))
                            st.balloons()
                        else:
                            st.warning(_("😔 Sorry, there is a system error, please try again later."))
                    else:
                        st.warning(_("😔 Sorry, this stock is not available on Market, is it a typo?"))

    with crypto_col:
        search_crypto_symbol = st.text_input(_("Search Crypto"), key="search_crypto", help=_("A decent crypto should have Profit/Loss ratio of 1 or more"))
        
        if search_crypto_symbol:
            if search_stock(search_crypto_symbol) != None:
                with st.container():
                    toggle_slider = st.slider(_("💰 Investment Amount ($)"), min_value=100, max_value=10000, value=100, step=1000, key=f"toggle_{search_crypto_symbol}")
                    # Display metrics for each period in columns
                    expected_loss, expected_profit, profit_loss_ratio = find_expected_value(search_crypto_symbol, 'crypto', 3)
                    # Expected Earnings        
                    expected_earnings = expected_profit / 100 * toggle_slider

                    if profit_loss_ratio >= 1:
                        expected_loss = expected_loss / 100 * toggle_slider
                        st.markdown(f"""
                            <div style='display: flex; justify-content: space-around;'>
                                <div style='color: #4CAF50; text-align: center;'>
                                    <p style='font-size:24px; font-weight:bold'> {_('Expected Earnings')}: ${expected_earnings:+,.2f}</p>
                                </div>
                                <div style='color: #4CAF50; text-align: center;'>
                                    <p style='font-size:24px; font-weight:bold'>{_('BUY')} 📈</p>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        expected_loss = expected_loss / 100 * toggle_slider
                        st.markdown(f"""
                            <div style='display: flex; justify-content: space-around;'>
                                <div style='color: #FF0000; text-align: center;'>
                                    <p style='font-size:24px; font-weight:bold'>{_('Expected Loss')}: ${expected_loss:+,.2f}</p>
                                </div>
                                <div style='color: #FF0000; text-align: center;'>
                                    <p style='font-size:24px; font-weight:bold'>{_('SELL')} 📉</p>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.error(f"{_('Crypto')} {search_crypto_symbol} {_('not found, do you want to contribute this crypto to our database?')}")
                if st.button(_("✨ Contribute New Crypto"), use_container_width=True):
                    if check_crypto_exists(search_crypto_symbol.upper()):
                        ticker, full_name = check_crypto_exists(search_crypto_symbol)
                        if add_crypto_to_database(ticker, full_name):
                            st.success(_("🎉 Thanks for your contribution! Your crypto will be added to the database tomorrow."))
                            st.balloons()
                        else:
                            st.warning(_("😔 Sorry, this crypto is not available on Market, is it a typo?"))

def portfolio_section():
    # Create a Space by markdown
    st.markdown("""
        <div style="height: 20px;"></div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
            <div style="text-align: center; font-size: 24px; font-weight: bold; color: #2c3e50;">
                {}
            </div>
        """.format(_("Portfolio Analytics Report")), unsafe_allow_html=True)

        st.markdown("""
            <style>
                .css-1y0tads {padding-top: 0px; padding-bottom: 0px;}
            </style>
        """, unsafe_allow_html=True)

        # Create a fixed container at the bottom for metrics
        with st.container():
            if existing_portfolio(st.session_state['username']):
                # Compute the metrics
                metrics = compute_portfolio_metrics(st.session_state['username'])
                
                # Layout the metrics
                col1, col2, col3 = st.columns(3, gap="small")
                
                # Display the metrics
                with col1:
                    st.markdown(f"""
                        <div style='text-align: center'>
                            <div style='font-size: 0.8em; color: #666'>😈 Greed Index</div>
                            <div style='font-size: 1.8em; font-weight: bold; color: {'#2ecc71' if metrics['greed_index'] < 50 else '#e74c3c'}'>{metrics['greed_index']:,.0f}/100</div>
                        </div>
                        """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                        <div style='text-align: center'>
                            <div style='font-size: 0.8em; color: #666'>⚠️ Risk Index</div>
                            <div style='font-size: 1.8em; font-weight: bold; color: {'#2ecc71' if metrics['risk_index'] < 33 else '#f39c12' if metrics['risk_index'] < 66 else '#e74c3c'}'>{metrics['risk_index']:,.0f}/100</div>
                        </div>
                        """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                        <div style='text-align: center'>
                            <div style='font-size: 0.8em; color: #666'>📈 Profit/Loss</div>
                            <div style='font-size: 1.8em; font-weight: bold; color: {'#2ecc71' if metrics['overall_return'] > 0 else '#e74c3c'}'>{metrics['overall_return']:,.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
            # Create a container for the chart
            with st.container():
                user_portfolio_chart = portfolio_chart(st.session_state['username'])
                if not user_portfolio_chart:
                    st.markdown("""
                        <div style="text-align: center; font-size: 20px; font-weight:normal; color: green;">
                            {}
                            <br>
                            {}
                        </div>
                    """.format(
                        _("Dear user, you don't have any portfolio yet, you can add one by navigating to EDIT PORTFOLIO in the sidebar."),
                        _("We will leverage data-driven approach and AI to monitor and analyze your portfolio in Real-Time!!")), unsafe_allow_html=True)
            
            # Create a Space by markdown
            st.markdown("""
                <div style="height: 20px;"></div>
            """, unsafe_allow_html=True)

def stock_pick_section(cur_alert_dict):
    # Create a container for the alerts section
    alert_container = st.container()

    with alert_container:           
        # Display header
        st.markdown("""
            <div style="text-align: center; font-size: 24px; font-weight: bold; color: #2c3e50;">
                {}
            </div>
        """.format(_("CondVest Pick 💰")), unsafe_allow_html=True)
        
        # Display alert section with title and results
        def display_alert_section(title, results, badge_color):
            # Add tooltip css 
            st.markdown(tooltip_css, unsafe_allow_html=True)
            # Display title with reduced size and spacing
            message = _("Worth a Try 💰") if title == _("💰 Buy!") else _("Could be a bounce this week 💰")
            st.markdown(f"""
                        <div style="text-align: center; margin: 5px 0;">
                            <div class="tooltip" style="font-size: 20px; font-weight: bold; color: {badge_color};">
                                {title}
                                <span class="tooltiptext">{message}</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

            if not results:
                st.markdown("""
                    <div style="text-align: center; font-size: 14px; font-weight: bold; color: grey; min-height: 50px; display: flex; align-items: center; justify-content: center;">
                        {}
                    </div>
                """.format(_("No Opportunity found today, bored... 😴")), unsafe_allow_html=True)
            else:
                container_class = "buy-container" if badge_color == "#4CAF50" else "sell-container"
                badge_class = "buy-badge" if badge_color == "#4CAF50" else "sell-badge"                        
                # Add tooltip css
                st.markdown(tooltip_css, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <style>
                        .{container_class} {{
                            display: flex;
                            flex-wrap: wrap;
                            gap: 5px;
                            min-height: 5px;
                            align-items: center;
                            justify-content: center;
                            padding: 5px;
                        }}
                        .{badge_class} {{
                            background-color: {badge_color} !important;
                            color: white;
                            padding: 4px 8px;
                            border-radius: 4px;
                            font-size: 14px;
                            text-align: center;
                        }}
                    </style>
                """, unsafe_allow_html=True)

                st.markdown(f'<div class="{container_class}">', unsafe_allow_html=True)
                for symbol in results:
                    st.markdown(f'<div class="{badge_class}">{symbol}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        acc_alert_col, main_alert_col = st.columns(2)
        acc_alert_symbols = find_alert_symbols(cur_alert_dict, 'ext_long_accelerating')
        main_alert_symbols = find_alert_symbols(cur_alert_dict, 'ext_accumulating')
        with acc_alert_col:
            display_alert_section(_("💰 Buy!"), acc_alert_symbols, "#4CAF50")
        with main_alert_col:
            display_alert_section(_("💰 Possible Bounce!"), main_alert_symbols, "#FFA500")
        
        if acc_alert_symbols or main_alert_symbols:
            st.session_state['alert_symbols'] = acc_alert_symbols + main_alert_symbols
        else:
            st.session_state['alert_symbols'] = []

    # Create Space by markdown
    st.markdown("""
        <div style="height: 20px;"></div>
    """, unsafe_allow_html=True)

def user_section(user_col, current_alerts_dict):
    # Stock Pick Section
    stock_pick_section(current_alerts_dict)
    # Symbol Exp Return Section
    symbol_exp_return_section(user_col)
    # Portfolio Section
    portfolio_section()
    
def market_section(main_col):
    # Create a container for the overview section
    main_container = st.container()
    with main_container:

        st.markdown("""
            <div style="text-align: center; font-size: 24px; font-weight: bold; color: #2c3e50;">
                {}
            </div>
        """.format(_('Market Health Overview'))
        , unsafe_allow_html=True)
        # Create columns for the instrument dropdown, symbol selection, and chart type
        instrument_col, symbol_col, chart_type_col = main_col.columns([1, 1, 1]) 

        # Dropdown for instrument selection
        with instrument_col:
            # Mapping of internal values to display values
            instrument_mapping = {
                "index": _("Index"),
                "commodity": _("Commodity"),
                "sector": _("Sector"),
                "bond": _("Bond")
            }
            
            # Reverse mapping for lookup
            display_to_internal = {v: k for k, v in instrument_mapping.items()}
            
            # Display the selectbox with translated options
            selected_display_value = st.selectbox(
                _("Select Instrument"),
                options=list(instrument_mapping.values()),
                key="instrument_dropdown"
            )
            
            # Get the internal value based on the selected display value
            selected_instrument = display_to_internal[selected_display_value]
            st.session_state['instrument'] = selected_instrument  # Use the internal value
            
        # Symbol selection dropdown
        with symbol_col:
            if 'instrument' not in st.session_state:
                st.session_state['instrument'] = "index"
            try:
                symbols = fetch_data(st.session_state['instrument'], 1)['symbol'].unique()
                # Mapping symbols to descriptive names
                if st.session_state['instrument'] == "index":
                    symbols_mapping = {
                        "^IXIC": _("Nasdaq Composite Index"),
                        "^DJI": _("Dow Jones Industrial Average"),
                        "^RUT": _("Russell 2000 Index"),
                        "^GSPC": _("S&P 500 Index")
                    }
                elif st.session_state['instrument'] == "commodity":
                    symbols_mapping = {
                        "GC=F": _("Gold Futures"),
                        "NG=F": _("Natural Gas Futures"),
                        "CL=F": _("Crude Oil Futures"),
                        "SI=F": _("Silver Futures")
                    }
                elif st.session_state['instrument'] == "bond":
                    symbols_mapping = {
                        "^TNX": _("10-Year Treasury Yield"),
                        "^TYX": _("30-Year Treasury Yield"),
                        "^FVX": _("5-Year Treasury Yield")
                    }
                elif st.session_state['instrument'] == "sector":
                    symbols_mapping = {
                        "SOXX": _("iShares Semiconductor ETF"),
                        "XLK": _("Technology Select Sector SPDR"),
                        "XLV": _("Health Care Select Sector SPDR"),
                        "XLF": _("Financial Select Sector SPDR"),
                        "XLE": _("Energy Select Sector SPDR"),
                        "XLB": _("Materials Select Sector SPDR"),
                        "XLRE": _("Real Estate Select Sector SPDR"),
                        "XLU": _("Utilities Select Sector SPDR"),
                        "XLY": _("Consumer Discretionary Select Sector SPDR"),
                        "XLP": _("Consumer Staples Select Sector SPDR")
                    }

                # Fetch symbols (dummy list for example purposes)
                symbols = list(symbols_mapping.keys())
                # Display selectbox with mapped options
                selected_symbol = st.selectbox(_("Select Symbol"), options=symbols, format_func=lambda x: symbols_mapping.get(x, x))
            except Exception as e:
                st.error(f"No symbols found for this instrument, please select another one.")
                return
        
        # Chart type radio buttons
        with chart_type_col:
            # Mapping of internal values to display values
            chart_type_mapping = {
                "Line Chart": _("Line Chart"),
                "Cumulative Return": _("Cumulative Return"),
                "Candlesticks": _("Candlesticks")
            }
            # Reverse mapping for lookup
            display_to_internal = {v: k for k, v in chart_type_mapping.items()}
            
            chart_type = st.selectbox(_("Chart Type"), 
                                    options=list(chart_type_mapping.values()), 
                                    key="chart_type")
            
            selected_chart_type = display_to_internal[chart_type]
            
        # Display the chart below the selection section
        intuitive_col, chart_col = st.columns([1, 3])
        
        with intuitive_col:
            interval_mapping = {
                        1: _("Short Term"),
                        3: _("Short-Medium Term"),
                        5: _("Medium Term"),
                        8: _("Medium-Long Term"),
                        13: _("Long Term")
                        }
            # Fetch alert data
            try:
                alert_data = fetch_alert_data(st.session_state['instrument'], selected_symbol)
            except Exception as e:
                st.error(_("No data found for this instrument, please select another one."))
                return
            
            # Fetch distinct intervals
            try:
                distinct_intervals = alert_data['interval'].unique()
                interval_labels = [interval_mapping[interval] for interval in distinct_intervals]
                label_to_interval = {v: k for k, v in interval_mapping.items()}
                
            except KeyError as e:
                st.error(_("No data found for this instrument, please select another one."))
                return
            
            # Slidebar for interval selection
            selected_label = st.select_slider(" ", options=interval_labels, key='interval_slider',
                                            help=_("Keep in mind, the longer the term, the stronger the signal. Focusing on Short Term only is a bit risky 🤫."),
                                            label_visibility="collapsed")
            selected_interval = label_to_interval[selected_label]
            
            # Fetch the latest alert data for the selected interval
            alert_data = alert_data.sort_values(by='date', ascending=True)
            cur_alert_data = alert_data[alert_data['interval'] == selected_interval].iloc[-1]['alerts']
            
            # Function to get dot color based on alert data
            def get_dot_color(cur_alert_data):
                
                if "velocity_alert" in cur_alert_data:
                    if "velocity_maintained" in cur_alert_data['velocity_alert']['alert_type']:
                        return "green"  # Healthy signal
                    
                    elif "velocity_loss" in cur_alert_data['velocity_alert']['alert_type']:
                        return "red"  # Pessimistic signal
                    
                    elif "velocity_weak" in cur_alert_data['velocity_alert']['alert_type']:
                        return "orange"  # Moderate or risk signal
                    
                    elif "velocity_negotiating" in cur_alert_data['velocity_alert']['alert_type']:
                        return "yellow"  # Neutral signal
                return "grey"  # Default for consolidating
            
            # Update dot color based on the logic
            dot_colors = [
                get_dot_color(cur_alert_data) if "velocity_maintained" in cur_alert_data['velocity_alert'].get('alert_type', '') else "grey",
                get_dot_color(cur_alert_data) if "velocity_weak" in cur_alert_data['velocity_alert'].get('alert_type', '') else "grey",
                get_dot_color(cur_alert_data) if "velocity_loss" in cur_alert_data['velocity_alert'].get('alert_type', '') else "grey",
                get_dot_color(cur_alert_data) if "velocity_negotiating" in cur_alert_data['velocity_alert'].get('alert_type', '') else "grey",
                get_dot_color(cur_alert_data) if "velocity_maintained" not in cur_alert_data['velocity_alert'].get('alert_type', '') and
                "velocity_weak" not in cur_alert_data['velocity_alert'].get('alert_type', '') and
                "velocity_loss" not in cur_alert_data['velocity_alert'].get('alert_type', '') else "grey"
            ]
            
            # Display the dots with the correct color and uniform text alignment
            st.markdown(tooltip_css, unsafe_allow_html=True)
            st.markdown(f"""
                <div style="display: flex; flex-direction: column; align-items: flex-end; position: absolute; top: 20px; right: 45px; gap: 30px;">
                    <div class="tooltip" style="display: flex; flex-direction: row; align-items: center; gap: 10px;">
                        <div style="width: 15px; height: 15px; background-color: {dot_colors[0]}; border-radius: 50%;"></div> 
                        <div style="width: 100px; font-size: 20px; color: #333; font-weight: bold; text-align: left;">{_("Optimistic")}</div>
                        <span class="tooltiptext">{_("Market is maintaining its upward momentum")}</span>
                    </div>
                    <div class="tooltip" style="display: flex; flex-direction: row; align-items: center; gap: 10px;">
                        <div style="width: 15px; height: 15px; background-color: {dot_colors[1]}; border-radius: 50%;"></div>
                        <div style="width: 100px; font-size: 20px; color: #333; font-weight: bold; text-align: left;">{_("Neutral")}</div>
                        <span class="tooltiptext">{_("Market is slowing down")}</span>
                    </div>
                    <div class="tooltip" style="display: flex; flex-direction: row; align-items: center; gap: 10px;">
                        <div style="width: 15px; height: 15px; background-color: {dot_colors[2]}; border-radius: 50%;"></div>
                        <div style="width: 100px; font-size: 20px; color: #333; font-weight: bold; text-align: left;">{_("Pessimistic")}</div>
                        <span class="tooltiptext">{_("Market is losing its upward momentum")}</span>
                    </div>
                    <div class="tooltip" style="display: flex; flex-direction: row; align-items: center; gap: 10px;">
                        <div style="width: 15px; height: 15px; background-color: {dot_colors[3]}; border-radius: 50%;"></div>
                        <div style="width: 100px; font-size: 20px; color: #333; font-weight: bold; text-align: left;">{_("Consolidating")}  </div>
                        <span class="tooltiptext">{_("Market is consolidating")}</span>
                    </div>
                    
                </div>
            """, unsafe_allow_html=True)
        
        with chart_col:
            try:
                # Plot the chart
                chart = overview_chart(st.session_state['instrument'], selected_symbol, selected_chart_type, selected_interval)
                st.plotly_chart(chart, use_container_width=True)
            except Exception as e:
                st.error(_("System Error, please try again later."))
                return  
            
# AI Chatbox Section
def ai_section():
    # Load pre-configured system message into session state if not already loaded
    if "messages" not in st.session_state:
        st.session_state.messages = PRECONFIGURED_MESSAGE

    # Chat Header
    st.markdown("""
        <div class="chat-header" style="
            text-align: center;
            padding: 15px;
            background-color: #f0f8ff;
            border-radius: 10px;
            margin: 10px 0;
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            {}
        </div>
    """.format(_('CondVest AI Advisor Chat')), unsafe_allow_html=True)

    # User Input Section
    prompt = st.text_input(
        "",
        placeholder="Ask me how this works or about platform features.",
        help="Type your question here and press Enter.",
        label_visibility="collapsed"
    )

    if prompt:
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display User Message
        with st.chat_message("user"):
            st.markdown(f"""
                <div style="width: 100%; word-wrap: break-word;">
                    {prompt}
                </div>
            """, unsafe_allow_html=True)

        # Generate and Display Assistant Response
        try:
            with st.chat_message("assistant"):
                response = ai_client.chat.completions.create(
                    model="gpt-4o-mini",  # Replace with your model
                    messages=st.session_state.messages,
                    temperature=0.5,
                    max_tokens=2048,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )

                # Extract response content
                assistant_response = response.choices[0].message.content

                # Add assistant response to session state
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})

                # Display Assistant Message
                st.markdown(f"""
                    <div style="width: 100%; word-wrap: break-word;">
                        {assistant_response}
                    </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")

def user_dashboard():

    # Set up the gettext translation
    locale_dir = os.path.join(os.path.dirname(__file__), 'locale')
    lang = 'en'  # Default language
    # Function to initialize and update translations
    def set_translation(language):
        global _
        translation = gettext.translation(
            'messages',  # Domain
            localedir=locale_dir,
            languages=[language],
            fallback=True
        )
        translation.install()
        _ = translation.gettext
        
    # Scrolling message of sandbox testing results
    scrolling_message = analyze_strategy_results()
    
    # Prepare the scrolling message with the results
    st.markdown("""
                <style>
                /* Container for scrolling area */
                .marquee-container {
                    overflow: hidden;
                    white-space: nowrap;
                    background-color: #f0f8ff;
                    padding: 10px;
                    border-radius: 30px;
                    width: 100%;
                    box-sizing: border-box;
                }

                /* Text styling and animation */
                .marquee {
                    display: inline-block;
                    padding-left: 100%; /* Start outside view */
                    font-size: 18px;
                    font-weight: bold;
                    animation: scroll 30s linear infinite; 
                }

                /* Define the scrolling animation */
                @keyframes scroll {
                    100% { transform: translateX(100%); } /* Start closer */
                    100% { transform: translateX(-100%); }
                }
                </style>
        """, unsafe_allow_html=True)
    # Create scrolling text with metrics with colored values
    scrolling_text = (
        '{}: <span style="color: #4CAF50">{}%</span> | '
        '{}: <span style="color: #FF5733">{}%</span> | '
        '{}: <span style="color: #4CAF50">{}%</span> | '
        '{}: <span style="color: #FF5733">{}%</span> | '
    ).format(
        _('Average Profit'),
        scrolling_message['avg_profit'],
        _('Average Loss'),
        scrolling_message['avg_loss'],
        _('Best Trade'),
        scrolling_message['best_pick'],
        _('Worst Trade'),
        scrolling_message['worst_pick']
    )
    # Increment the view count each time the dashboard is accessed
    increment_dashboard_view_count()
    
    # Display scrolling marquee in Streamlit
    st.markdown(
        f"""
        <div class="marquee-container">
            <span class="marquee">
                {_("CondVest Alerts Performance")}: {scrolling_text}
                {_("Dashboard Views")}: {get_dashboard_view_count()}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Prepare data for display data
    most_recent_trade_date = pd.to_datetime(get_most_current_trading_date())
    candidate_collection = initialize_mongo_client()[DB_NAME][CANDIDATE_COLLECTION]
    current_alerts_dict = list(candidate_collection.find({"date": {"$gte": most_recent_trade_date}}))
    
    # Display the dashboard
    market_col, user_col = st.columns([3, 2])
    with market_col:
        # Market section
        market_section(market_col)
        # AI section
        ai_section()
        
    with user_col:
        # Display all content
        user_section(user_col, current_alerts_dict)
