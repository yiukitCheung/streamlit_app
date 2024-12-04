import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from pymongo import MongoClient
import yfinance as yf
import numpy as np
import redis, io, time
from datetime import datetime
from long_term import long_term_dashboard
from dependencies import search_stock, add_stock_to_database, check_symbol_yahoo
# Ensure the correct path to the 'data' directory
from analyzer import ExpectedReturnRiskAnalyzer
from add_portfolio import existing_portfolio
from config.mongdb_config import load_mongo_config
from openai import OpenAI

client = OpenAI(api_key=st.secrets['chatgpt']['api_key'])
thread = client.beta.threads.create()  # Create a thread for the conversation


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
    try:
        return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD)
    except Exception as e:
        st.error(f"Error initializing Redis client: {e}")
        return None

@st.cache_data
def analyze_strategy_results():
    # Fetch the results from MongoDB (simulate this for now)
    results_df = pd.DataFrame(list(initialize_mongo_client()[DB_NAME][SANDBOX_COLLECTION].find({})))
    
    # Ensure the profit/loss column is used correctly and already in decimal format
    results_df['profit_loss_pct'] = results_df['final_profit_loss_pct']  # Adjusted column name

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
        "best_trade": round(profits.max() * 100, 2),  # Convert to percentage
        "worst_trade": round(losses.min() * 100, 2),  # Convert to percentage
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
            data = pd.read_json(io.StringIO(cached_data.decode("utf-8")))
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
    
    collection_obj = initialize_mongo_client()[DB_NAME][ALERT_COLLECTION]
    query = {"instrument": instrument, "symbol": symbol}
    cursor = collection_obj.find(query)
    
    return pd.DataFrame(list(cursor))

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
            return "yellow", "Flag for Risk"  # Velocity weak 
        elif "velocity_negotiating" in alert_data['alert_type']:
            return "yellow", "Flag for Risk"  # Velocity negotiating
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
    # Display each stock in the portfolio and its alert status
    for symbol, details in portfolio.items():
        symbol = symbol.upper()
        today = get_most_current_trading_date()
        # Fetch the alert for the stock
        alert_data = alert_collection.find_one({'symbol': symbol, 'date':{'$gte': pd.to_datetime(today)}}, projection={'alerts': True, '_id': False})
        
        if alert_data and 'velocity_alert' in alert_data['alerts']:
            velocity_alert = alert_data['alerts'].get('velocity_alert', {})
            color, status = get_alert_color(velocity_alert)
        else:
            color, status = "white", "No Status"  # No alert or empty alert field
        # Display the stock symbol and the alert status with a colored dot in a scrollable container
        st.markdown(f"""
            <div style="max-height: 200px; overflow-y: auto; padding: 10px;">
                <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 10px;">
                    <div style="width: 20px; height: 20px; background-color: {color}; border-radius: 50%; margin-right: 10px;"></div>
                    <div style="font-size: 18px; font-weight: bold; color: #2c3e50;">{symbol.upper()} - {status}</div>
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
            range=[min(filtered_df.loc[:,'close']) * 0.9, max(filtered_df.loc[:,'close']) * 1.1],  # Adjust the range to add padding
            title='Value',
            showgrid=True,  # Optional: show gridlines for better readability
            zeroline=True  # Optional: show a zero line if needed
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

@st.cache_data
def find_expected_value(symbol: str, selected_period: int):
    # Find expected support and resistance
    find_expected_value = ExpectedReturnRiskAnalyzer()
    expected_support, expected_resistance = find_expected_value.find_sup_res(symbol.upper(), selected_period)
    # Find the ema support and resistance
    for entry in expected_support:
        if entry == 'emas' and expected_support[entry] != float('-inf'):
            support = expected_support[entry]
            break
        elif entry == 'dense_area' and expected_support[entry] != float('-inf'):
            support = expected_support[entry]
            break
        elif entry == 'fibonacci' and expected_support[entry] != float('-inf'):
            support = expected_support[entry]
            break
        
    for entry in expected_resistance:
        if entry == 'emas' and expected_resistance[entry] != float('inf'):
            resistance = expected_resistance[entry]
            break
        elif entry == 'dense_area' and expected_resistance[entry] != float('inf'):
            resistance = expected_resistance[entry]
            break
        elif entry == 'fibonacci' and expected_resistance[entry] != float('inf'):
            resistance = expected_resistance[entry]
            break
    
    cur_price = fetch_data('equity', 1).loc[fetch_data('equity', 1)['symbol'] == symbol.upper()].iloc[-1]['close']
    expected_loss = round((support - cur_price) / cur_price, 2)  * 100
    expected_profit = round((resistance - cur_price) / cur_price, 2) * 100
    profit_loss_ratio = abs(round(expected_profit / expected_loss, 2))
    return expected_loss, expected_profit, profit_loss_ratio

@st.cache_data
def compute_portfolio_metrics(username: str):
    # Initialize session state variables if they don't exist
    if 'overall_return' not in st.session_state:
        st.session_state['overall_return'] = 0
        st.session_state['previous_overall_return'] = 0
    if 'greed_index' not in st.session_state:
        st.session_state['greed_index'] = 0
        st.session_state['previous_greed_index'] = 0
    if 'risk_index' not in st.session_state:
        st.session_state['risk_index'] = 0
        st.session_state['previous_risk_index'] = 0

    # Compute Overall Return
    user_portfolio = initialize_mongo_client()[DB_NAME][PORTFOLIO_COLLECTION].find_one({'username': username}, projection={'portfolio': True, '_id': False})
    if not user_portfolio or 'portfolio' not in user_portfolio or not user_portfolio['portfolio']:
        return False
        
    portfolio_df = pd.DataFrame(user_portfolio['portfolio']).T
    symbol_list = portfolio_df.index.tolist()
    
    # Get current prices for all symbols
    current_prices = fetch_data('equity', 1)
    current_prices = current_prices[current_prices['symbol'].isin(symbol_list)]
    current_prices = current_prices.groupby('symbol')['close'].last()
    
    # Add current prices to portfolio dataframe
    portfolio_df['current_close'] = portfolio_df.index.map(current_prices)
    
    # Convert string values to numeric if needed
    portfolio_df['shares'] = pd.to_numeric(portfolio_df['shares'])
    portfolio_df['avg_price'] = pd.to_numeric(portfolio_df['avg_price'])
    portfolio_df['current_close'] = pd.to_numeric(portfolio_df['current_close'])
    
    # Calculate metrics
    portfolio_df['total_cost'] = portfolio_df['shares'] * portfolio_df['avg_price']
    portfolio_df['current_value'] = portfolio_df['shares'] * portfolio_df['current_close']
    portfolio_df['return'] = (portfolio_df['current_value'] - portfolio_df['total_cost']) / portfolio_df['total_cost']
    portfolio_df['return'] = portfolio_df['return'].apply(lambda x: round(x, 2))
    
    overall_return = portfolio_df['return'].mean()
    
    # Compute the greed index
    # Using logarithmic function to map gains to greed index
    # Formula: greed_index = 50 * log(1 + 2 * gain) / log(3)
    greed_index = min(100, round(50 * np.log(1 + 2 * abs(overall_return)) / np.log(3), 2))
    
    # Compute the risk index
    # Fetch the possible downside percentage
    total_possible_downside = 0
    for symbol in symbol_list:
        possible_downside = find_expected_value(symbol, 1)[0]
        total_possible_downside += portfolio_df.loc[symbol, 'shares'] * possible_downside
    
    risk_index = total_possible_downside / portfolio_df['total_cost'].sum()
    
    # Update session state
    st.session_state['previous_overall_return'] = st.session_state['overall_return']
    st.session_state['previous_greed_index'] = st.session_state['greed_index']
    st.session_state['previous_risk_index'] = st.session_state['risk_index']
    
    st.session_state['overall_return'] = overall_return
    st.session_state['greed_index'] = greed_index
    st.session_state['risk_index'] = risk_index

def display_user_dashboard_content(cur_alert_dict=None):
    
    # Create a container for the overview section
    main_container = st.container()
    with main_container:
        col_overview, col_portfolio = st.columns([2, 1.5])
        with col_overview:
            overview_container = st.container()
            with overview_container:
                st.markdown("""
                    <div style="text-align: center; font-size: 24px; font-weight: bold; color: #2c3e50;">
                        Market Health Overview
                    </div>
                """, unsafe_allow_html=True)
                # Create columns for the instrument dropdown, symbol selection, and chart type
                col1, col2, col3 = st.columns([1, 1, 1]) 

                # Dropdown for instrument selection
                with col1:
                    instrument_options = ["Index", "Commodity", "Sector", "Bond"]
                    selected_instrument = st.selectbox("Select Instrument", options=instrument_options, key="instrument_dropdown")
                    st.session_state['instrument'] = selected_instrument.lower()  # Set session state based on selection
                # Symbol selection dropdown
                with col2:
                    if 'instrument' not in st.session_state:
                        st.session_state['instrument'] = "index"
                    try:
                        symbols = fetch_data(st.session_state['instrument'], 1)['symbol'].unique()
                        # Mapping symbols to descriptive names
                        if st.session_state['instrument'] == "index":
                            symbols_mapping = {
                                "^IXIC": "Nasdaq Composite Index",
                                "^DJI": "Dow Jones Industrial Average",
                                "^RUT": "Russell 2000 Index",
                                "^GSPC": "S&P 500 Index"
                            }
                        elif st.session_state['instrument'] == "commodity":
                            symbols_mapping = {
                                "GC=F": "Gold Futures",
                                "NG=F": "Natural Gas Futures",
                                "CL=F": "Crude Oil Futures",
                                "SI=F": "Silver Futures"
                            }
                        elif st.session_state['instrument'] == "bond":
                            symbols_mapping = {
                                "^TNX": "10-Year Treasury Yield",
                                "^TYX": "30-Year Treasury Yield",
                                "^FVX": "5-Year Treasury Yield"
                            }
                        elif st.session_state['instrument'] == "sector":
                            symbols_mapping = {
                                "SOXX": "iShares Semiconductor ETF",
                                "XLK": "Technology Select Sector SPDR",
                                "XLV": "Health Care Select Sector SPDR",
                                "XLF": "Financial Select Sector SPDR",
                                "XLE": "Energy Select Sector SPDR",
                                "XLB": "Materials Select Sector SPDR",
                                "XLRE": "Real Estate Select Sector SPDR",
                                "XLU": "Utilities Select Sector SPDR",
                                "XLY": "Consumer Discretionary Select Sector SPDR",
                                "XLP": "Consumer Staples Select Sector SPDR"
                            }

                        # Fetch symbols (dummy list for example purposes)
                        symbols = list(symbols_mapping.keys())
                        
                        # Display selectbox with mapped options
                        selected_symbol = st.selectbox("Select Symbol", options=symbols, format_func=lambda x: symbols_mapping.get(x, x))
                    except Exception as e:
                        st.error(f"Error fetching symbols: {str(e)}")
                # Chart type radio buttons
                with col3:
                    chart_type = st.selectbox("Chart Type", options=["Line Chart", "Cumulative Return", "Candlesticks"], key="chart_type")
                
                # Display the chart below the selection section
                chart_layout = st.columns([1, 3])
                
                with chart_layout[0]:
                    interval_mapping = {
                                1: "Short Term",
                                3: "Short-Medium Term",
                                5: "Medium Term",
                                8: "Medium-Long Term",
                                13: "Long Term"}
                    
                    alert_data = fetch_alert_data(st.session_state['instrument'], selected_symbol)
                    distinct_intervals = alert_data['interval'].unique()
                    interval_labels = [interval_mapping[interval] for interval in distinct_intervals]
                    label_to_interval = {v: k for k, v in interval_mapping.items()}

                    # Slidebar for interval selection
                    selected_label = st.select_slider(" ", options=interval_labels, key='interval_slider', help="Keep in mind, the longer the term, the stronger the signal. Focusing on Short Term only is a bit risky 🤫.")
                    selected_interval = label_to_interval[selected_label]
                    cur_alert_data = alert_data[alert_data['interval'] == selected_interval].iloc[-1]['alerts']
                    
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
                                <div style="width: 100px; font-size: 20px; color: #333; font-weight: bold; text-align: left;">Optimistic</div>
                                <span class="tooltiptext">Market is maintaining its upward momentum</span>
                            </div>
                            <div class="tooltip" style="display: flex; flex-direction: row; align-items: center; gap: 10px;">
                                <div style="width: 15px; height: 15px; background-color: {dot_colors[1]}; border-radius: 50%;"></div>
                                <div style="width: 100px; font-size: 20px; color: #333; font-weight: bold; text-align: left;">Neutral</div>
                                <span class="tooltiptext">Market is slowing down</span>
                            </div>
                            <div class="tooltip" style="display: flex; flex-direction: row; align-items: center; gap: 10px;">
                                <div style="width: 15px; height: 15px; background-color: {dot_colors[2]}; border-radius: 50%;"></div>
                                <div style="width: 100px; font-size: 20px; color: #333; font-weight: bold; text-align: left;">Pessimistic</div>
                                <span class="tooltiptext">Market is losing its upward momentum</span>
                            </div>
                            <div class="tooltip" style="display: flex; flex-direction: row; align-items: center; gap: 10px;">
                                <div style="width: 15px; height: 15px; background-color: {dot_colors[3]}; border-radius: 50%;"></div>
                                <div style="width: 100px; font-size: 20px; color: #333; font-weight: bold; text-align: left;">Consolidating</div>
                                <span class="tooltiptext">Market is consolidating</span>
                            </div>
                            
                        </div>
                    """, unsafe_allow_html=True)
                with chart_layout[1]:
                    try:
                        chart = overview_chart(st.session_state['instrument'], selected_symbol, chart_type, selected_interval)

                        st.plotly_chart(chart, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error displaying chart: {str(e)}")
                        
        with col_portfolio:
            st.markdown("""
                <div style="text-align: center; font-size: 24px; font-weight: bold; color: #2c3e50;">
                    Portfolio Analytics Report
                </div>
            """, unsafe_allow_html=True)

            st.markdown("""
                <style>
                    .css-1y0tads {padding-top: 0px; padding-bottom: 0px;}
                </style>
            """, unsafe_allow_html=True)
            portfolio_container = st.container()
            with portfolio_container:
                # Create a fixed container at the bottom for metrics
                metrics_container = st.container()
                with metrics_container:
                    if existing_portfolio(st.session_state['username']):
                        # Compute the metrics
                        compute_portfolio_metrics(st.session_state['username'])
                        
                        # Layout the metrics
                        col1, col2, col3 = st.columns(3, gap="small")
                        
                        # Fetch the delta for the greed index
                        greed_delta = st.session_state['greed_index'] - st.session_state['previous_greed_index'] \
                            if 'previous_greed_index' in st.session_state else 0
                        risk_delta = st.session_state['risk_index'] - st.session_state['previous_risk_index'] \
                            if 'previous_risk_index' in st.session_state else 0
                        return_delta = st.session_state['overall_return'] - st.session_state['previous_overall_return'] \
                            if 'previous_overall_return' in st.session_state else 0
                        
                        # Display the metrics
                        with col1:
                            st.metric(
                                label="😈 Greed Index",
                                value=f"{st.session_state['greed_index']:,.2f}%",
                                delta=f"{greed_delta:+,.2f}%",
                                delta_color="inverse"
                            )
                        with col2:
                            st.metric(
                                label="⚠️ Risk Index",  
                                value=f"{st.session_state['risk_index']:,.2f}%",
                                delta=f"{risk_delta:+,.2f}%",
                                delta_color="normal"
                            )

                        with col3:
                            st.metric(
                                label = "📈 Profit/Loss",
                                value=f"{st.session_state['overall_return']:,.2f}%", 
                                delta=f"{return_delta:+,}%",
                                delta_color="inverse"
                            )
                # Create a container for the chart
                chart_container = st.container()
                with chart_container:
                    user_portfolio_chart = portfolio_chart(st.session_state['username'])
                    if not user_portfolio_chart:
                        st.markdown("""
                            <div style="text-align: center; font-size: 24px; font-weight: bold; color: grey;">
                                No Portfolio Data Found
                            </div>
                        """, unsafe_allow_html=True)
    user_exp_profit_loss_placeholder = st.empty()
    with user_exp_profit_loss_placeholder:
        col_exp_profit_loss, col_alert_section = st.columns([4, 3])
        
        with col_exp_profit_loss:
            st.markdown("""
                    <div style="text-align: center; font-size: 24px; font-weight: bold; color: #2c3e50;">
                        Check The Profitability of Stock Here 🔍
                    </div>
            """, unsafe_allow_html=True) 
            with st.container():
                col1, col2 = st.columns(2)
                
                with col1:
                    search_stock_symbol = st.text_input("Search Stock", key="search_stock")
                    
                with col2:
                    period_options = ["Short", "Medium", "Long"]
                    period_mapping = {
                        "Short": 3,
                        "Medium": 5,
                        "Long": 8
                    }
                    selected_period = st.selectbox("Period", options=period_options, key="exp_period")
                    selected_period_value = period_mapping[selected_period] 
            
                if search_stock_symbol:
                    if search_stock(search_stock_symbol) != None:
                        with st.container():
                            expected_loss, expected_profit, profit_loss_ratio = find_expected_value(search_stock_symbol, selected_period_value)
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                color = "#4CAF50" if expected_profit > abs(expected_loss) and (expected_profit > 5 )else "#FF0000"
                                st.markdown(f"""
                                    <div style='color: {color}'>
                                        <p style='font-size:20px; margin-bottom:0'>Expected Return/Loss</p>
                                        <p style='font-size:24px; font-weight:bold'>{expected_profit:+,}%</p>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                            with col2:
                                color = "#4CAF50" if (expected_profit > abs(expected_loss)) and (expected_profit > 5) else "#FF0000"
                                st.markdown(f"""
                                    <div style='color: {color}'>
                                        <p style='font-size:20px; margin-bottom:0'>Expected Risk</p>
                                        <p style='font-size:24px; font-weight:bold'>{expected_loss:+,}%</p>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                            with col3:
                                color = "#4CAF50" if profit_loss_ratio > 1 and (profit_loss_ratio > 5) else "#808080"
                                st.markdown(f"""
                                    <div style='color: {color}'>
                                        <p style='font-size:20px; margin-bottom:0'>Profit/Loss Ratio</p>
                                        <p style='font-size:24px; font-weight:bold'>{profit_loss_ratio:+,}</p>
                                    </div>
                                """, unsafe_allow_html=True)
                        
                    else:
                        st.error(f"Stock {search_stock_symbol} not found, do you want to contribute this stock to our database?")
                        if st.button("✨ Contribute New Stock", use_container_width=True):
                            if check_symbol_yahoo(search_stock_symbol.upper()):
                                search_stock_symbol = search_stock_symbol.upper()
                                full_name = yf.Ticker(search_stock_symbol).info.get('longName')
                                add_stock_to_database(search_stock_symbol, full_name)
                                st.success("🎉 Thanks for your contribution! Your stock will be added to the database tomorrow.")
                                st.balloons()
                            else:
                                st.warning("😔 Sorry, this stock is not available on Market, is it a typo?")

        with col_alert_section:         
            alert_container = st.container()

            with alert_container:           
                # Display header
                st.markdown("""
                    <div style="text-align: center; font-size: 24px; font-weight: bold; color: #2c3e50;">
                        Opportunity Alerts 💰
                    </div>
                """, unsafe_allow_html=True)

                # Add term selection dropdown
                selected_term = st.selectbox(
                    "Select Trading Term",
                    ["Short Term", "Mid Term", "Long Term"],
                    key="term_selector"
                )

                # Display alert section with title and results
                def display_alert_section(title, results, badge_color):
                    # Add tooltip css
                    st.markdown(tooltip_css, unsafe_allow_html=True)
                    # Display title
                    message = "The buy is suggesting to buy at tomorrow's open price" if title == "💰💰💰 Buy!" else "The possible bounce is suggesting to buy at tomorrow's open price"
                    st.markdown(f"""
                                <div class="centered-container">
                                    <div class="tooltip" style="font-size: 24px; font-weight: bold; color: {badge_color};">
                                        {title}
                                        <span class="tooltiptext">{message}</span>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

                    if not results:
                        st.markdown("""
                            <div style="text-align: center; font-size: 14px; font-weight: bold; color: grey; min-height: 100px; display: flex; align-items: center; justify-content: center;">
                                No Opportunity found today, bored... 😴
                            </div>
                        """, unsafe_allow_html=True)
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
                                    gap: 10px;
                                    min-height: 10px;
                                    align-items: center;
                                    justify-content: center;
                                }}
                                .{badge_class} {{
                                    background-color: {badge_color} !important;
                                    color: white;
                                    padding: 8px 12px;
                                    border-radius: 5px;
                                    font-size: 16px;
                                    text-align: center;
                                }}
                            </style>
                        """, unsafe_allow_html=True)

                        st.markdown(f'<div class="{container_class}">', unsafe_allow_html=True)
                        for symbol in results:
                            st.markdown(f'<div class="{badge_class}">{symbol}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                # Display alerts based on selected term
                if selected_term == "Short Term":
                    acc_alert = 'acclerating'
                    main_alert = 'main_accumulating'
                elif selected_term == "Mid Term":
                    acc_alert = 'long_accumulating'
                    main_alert = 'long_main_accumulating'
                else:  # Long Term
                    acc_alert = 'ext_long_accelerating'
                    main_alert = 'ext_accumulating'

                acc_alert_col, main_alert_col = st.columns(2)
                acc_alert_symbols = find_alert_symbols(cur_alert_dict, acc_alert)
                main_alert_symbols = find_alert_symbols(cur_alert_dict, main_alert)
                with acc_alert_col:
                    display_alert_section("💰 Buy!", acc_alert_symbols, "#4CAF50")
                with main_alert_col:
                    display_alert_section("💰 Possible Bounce!", main_alert_symbols, "#FFA500")
                
                if acc_alert_symbols or main_alert_symbols:
                    st.session_state['alert_symbols'] = acc_alert_symbols + main_alert_symbols
                else:
                    st.session_state['alert_symbols'] = []

def user_dashboard():
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
        f'Average Profit: <span style="color: #4CAF50">{scrolling_message["avg_profit"]}%</span> | '
        f'Average Loss: <span style="color: #FF5733">{scrolling_message["avg_loss"]}%</span> | '
        f'Best Trade: <span style="color: #4CAF50">{scrolling_message["best_trade"]}%</span> | '
        f'Worst Trade: <span style="color: #FF5733">{scrolling_message["worst_trade"]}%</span> | '

    )
    # Display scrolling marquee in Streamlit
    st.markdown(
        f"""
        <div class="marquee-container">
            <span class="marquee">
                CondVest Alerts Performance: {scrolling_text}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Prepare data for display data
    most_recent_trade_date = pd.to_datetime(get_most_current_trading_date())
    candidate_collection = initialize_mongo_client()[DB_NAME][CANDIDATE_COLLECTION]
    current_alerts_dict = list(candidate_collection.find({"date": {"$gte": most_recent_trade_date}}))
    
    # Display all content
    display_user_dashboard_content(cur_alert_dict=current_alerts_dict)
