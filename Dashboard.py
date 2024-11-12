import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from pymongo import MongoClient
import yfinance as yf
import numpy as np
from datetime import datetime
from long_term import long_term_dashboard
# Ensure the correct path to the 'data' directory
from stock_candidates_analysis import DailyTradingStrategy, Pick_Stock
from add_portfolio import existing_portfolio
# MongoDB Configuration
DB_NAME = st.secrets['mongo']['db_name']
WAREHOUSE_INTERVAL = st.secrets['mongo']['warehouse_interval']
WAREHOUSE_INTERVAL_COLLECTION = '1d_data'
PROCESSED_COLLECTION = st.secrets['mongo']['processed_collection_name']
ALERT_COLLECTION = st.secrets['mongo']['alert_collection_name']
CANDI_COLLECTION = st.secrets['mongo']['candidate_collection_name']
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
    client = MongoClient(st.secrets["mongo"]["host"])
    return client

def analyze_strategy_results(results):

    # Separate winning and losing trades based on final_profit_loss_pct
    profits = [trade['final_profit_loss_pct'] for trade in results if trade['final_profit_loss_pct'] > 0]
    losses = [trade['final_profit_loss_pct'] for trade in results if trade['final_profit_loss_pct'] <= 0]
    
    # Calculate win rate
    total_trades = len(results)
    win_trades = len(profits)
    win_rate = win_trades / total_trades if total_trades > 0 else 0

    # Calculate average profit and average loss
    avg_profit = sum(profits) / len(profits) if profits else 0
    avg_loss = sum(losses) / len(losses) if losses else 0

    # Calculate maximum drawdown (the largest peak-to-trough decline)
    cumulative_returns = [1 + trade['final_profit_loss_pct'] for trade in results]
    peak = cumulative_returns[0]
    max_drawdown = 0

    for value in cumulative_returns:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # Return results in a dictionary
    return {
        "win_rate": round(win_rate * 100, 2),  # Convert to percentage
        "avg_profit": round(avg_profit * 100, 2),  # Convert to percentage
        "avg_loss": round(avg_loss * 100, 2),  # Convert to percentage
        "max_drawdown": round(max_drawdown * 100, 2)  # Convert to percentage
    }
    
    
@st.cache_data
def fetch_data(instrument, interval):
    
    collection_obj = initialize_mongo_client()[DB_NAME][PROCESSED_COLLECTION]
    query = {"instrument": instrument, "interval": interval}
    cursor = collection_obj.find(query)
    
    return pd.DataFrame(list(cursor))

@st.cache_data
def fetch_alert_data(instrument, symbol):
    
    collection_obj = initialize_mongo_client()[DB_NAME][ALERT_COLLECTION]
    query = {"instrument": instrument, "symbol": symbol}
    cursor = collection_obj.find(query)
    
    return pd.DataFrame(list(cursor))

@st.cache_data
def fetch_return_data(instrument):
    if not WAREHOUSE_INTERVAL:
        raise ValueError("warehouse_interval is empty in st.secrets")

    # Fetch data from MongoDB for the specified symbol and date range
    df = fetch_data(instrument, 1)
    
    # Calculate cumulative returns based on the first available 'close' value
    distinct_symbols = df['symbol'].unique()
    symbol_dfs = []
    for symbol in distinct_symbols:
        symbol_df = df[df['symbol'] == symbol]
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

def overview_chart(instrument: str, selected_symbols: list, chart_type: str):
    # Plot the cumulative return of Index
    if chart_type == "Cumulative Return":
        cum_return_chart = go.Figure()
        cum_return_data = fetch_return_data(instrument)[['cumulative_return', 'date','symbol']]
        
        
        data = cum_return_data[cum_return_data['symbol'] == selected_symbols]
        
        cum_return_chart.add_trace(go.Scatter(
            x=data['date'],
            y=data['cumulative_return'],
            mode='lines+markers',
            name=selected_symbols,
            opacity=0.5
        ))
        
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
        df = fetch_data(instrument, 1)[['date', 'open', 'high', 'low', 'close', 'symbol']]
        
        # Filter the data for the selected symbol
        distinct_symbols = df['symbol'].unique() if not selected_symbols else selected_symbols
        filtered_df = df[df['symbol'] == distinct_symbols]

        candlestick_chart = go.Figure()
        candlestick_chart.add_trace(go.Candlestick(
            x=filtered_df['date'].astype(str),
            open=filtered_df['open'],
            high=filtered_df['high'],
            low=filtered_df['low'],
            close=filtered_df['close'],
            name='price'
        ))

        # Update layout to include range selector but without the range slider
        candlestick_chart.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=5, label="5y", step="year", stepmode="backward"),
                    ])
                )
            ),
            
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
        data = fetch_data(instrument, 1)[['close', 'date', 'symbol']]
        data = data[data['symbol'] == selected_symbols]

        line_chart.add_trace(go.Scatter(
            x=data['date'],
            y=data['close'],
            fill='tozeroy',
            name=selected_symbols,
            opacity=0.5
        ))
        
        # Update y-axis properties with padding
        line_chart.update_yaxes(
            range=[min(data['close']) * 0.9, max(data['close']) * 1.1],  # Adjust the range to add padding
            title='Value',
            showgrid=True,  # Optional: show gridlines for better readability
            zeroline=True  # Optional: show a zero line if needed
        )
        # Update layout to include range selector and range slider
        line_chart.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=5, label="5y", step="year", stepmode="backward"),
                        dict(step="all", label="Max")  # Display all available data
                    ])
                ),
                rangeslider=dict(visible=True),  # Add a range slider below the x-axis
                type="date"
            ),
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

def display_user_dashboard_content(cur_alert_dict=None):
    # Create a container for the overview section
    main_container = st.container()
    with main_container:
        col_overview, col_portfolio = st.columns([2, 1])
        with col_overview:
            overview_container = st.container()
            with overview_container:
                # Create columns for the instrument dropdown, symbol selection, and chart type
                col1, col2, col3 = st.columns([1, 1, 1])  # Adjust the width ratios as needed

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
                        selected_symbols = st.selectbox("Select Symbol", options=symbols)
                    except Exception as e:
                        st.error(f"Error fetching symbols: {str(e)}")

                # Chart type radio buttons
                with col3:
                    chart_type = st.selectbox("Chart Type", options=["Line Chart", "Cumulative Return", "Candlesticks"])
                    chart = overview_chart(st.session_state['instrument'], selected_symbols, chart_type)

                # Display the chart below the selection section
                chart_layout = st.columns([4, 1])
                with chart_layout[0]:
                    try:
                        st.plotly_chart(chart, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error displaying chart: {str(e)}")
                        
                with chart_layout[1]:
                    interval_mapping = {
                                1: "Short Term",
                                3: "Short-Medium Term",
                                5: "Medium Term",
                                8: "Medium-Long Term",
                                13: "Long Term"}
                    
                    alert_data = fetch_alert_data(st.session_state['instrument'], selected_symbols)
                    distinct_intervals = alert_data['interval'].unique()
                    interval_labels = [interval_mapping[interval] for interval in distinct_intervals]
                    label_to_interval = {v: k for k, v in interval_mapping.items()}

                    # Slidebar for interval selection
                    selected_label = st.select_slider(" ", options=interval_labels, key='interval_slider')
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
                    st.markdown(f"""
                        <div style="display: flex; flex-direction: column; align-items: flex-end; position: absolute; top: 15px; right: 0px; gap: 20px;">
                            <div style="display: flex; flex-direction: row; align-items: center; gap: 10px;">
                                <div style="width: 15px; height: 15px; background-color: {dot_colors[0]}; border-radius: 50%;"></div> 
                                <div style="width: 100px; font-size: 14px; color: #333; font-weight: bold; text-align: left;">Optimistic</div>
                            </div>
                            <div style="display: flex; flex-direction: row; align-items: center; gap: 10px;">
                                <div style="width: 15px; height: 15px; background-color: {dot_colors[4]}; border-radius: 50%;"></div> 
                                <div style="width: 100px; font-size: 14px; color: #333; font-weight: bold; text-align: left;">Neutral</div>
                            </div>
                            <div style="display: flex; flex-direction: row; align-items: center; gap: 10px;">
                                <div style="width: 15px; height: 15px; background-color: {dot_colors[1]}; border-radius: 50%;"></div>
                                <div style="width: 100px; font-size: 14px; color: #333; font-weight: bold; text-align: left;">Flag for Risk</div>
                            </div>
                            <div style="display: flex; flex-direction: row; align-items: center; gap: 10px;">
                                <div style="width: 15px; height: 15px; background-color: {dot_colors[2]}; border-radius: 50%;"></div>
                                <div style="width: 100px; font-size: 14px; color: #333; font-weight: bold; text-align: left;">Pessimistic</div>
                            </div>
                            <div style="display: flex; flex-direction: row; align-items: center; gap: 10px;">
                                <div style="width: 15px; height: 15px; background-color: {dot_colors[3]}; border-radius: 50%;"></div>
                                <div style="width: 100px; font-size: 14px; color: #333; font-weight: bold; text-align: left;">Consolidating</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

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
                        col1, col2, col3 = st.columns(3, gap="small")
                        number = np.random.randint(0, 100)
                        delta = np.random.randint(-10, 10)
                        with col1:
                            st.metric(
                                label="😈 Greed Index",
                                value=f"{number:,}%",
                                delta=f"{delta:+,}%",
                                delta_color="inverse"
                            )
                        number = np.random.randint(0, 100)
                        delta = np.random.randint(-10, 10)
                        with col2:
                            st.metric(
                                label="⚠️ Risk Index",  
                                value=f"{number:,}%",
                                delta=f"{delta:+,}",
                                delta_color="normal"
                            )
                        number = np.random.randint(0, 100)
                        delta = np.random.randint(-10, 10)
                        with col3:
                            st.metric(
                                label = "📈 Profit/Loss",
                                value=f"{number:,}%", 
                                delta=f"{delta:+,}",
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
            
    alert_container = st.container()
    with alert_container:           
        st.markdown("""
            <div style="text-align: center; font-size: 24px; font-weight: bold; color: #2c3e50;">
                Opportunity Alerts
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
            <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
                <div style="text-align: center; font-size: 18px; font-weight: bold; width: 33%;">Short Term</div>
                <div style="text-align: center; font-size: 18px; font-weight: bold; width: 33%;">Mid Term</div>
                <div style="text-align: center; font-size: 18px; font-weight: bold; width: 33%;">Long Term</div>
            </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        # Short Term Column
        with col1:
            results = find_alert_symbols(cur_alert_dict, 'accelerating')
            st.markdown("""
                <div style="text-align: center; font-size: 18px; font-weight: bold; color: #4CAF50;">
                    Accelerating 
                </div>
            """, unsafe_allow_html=True)
            if not results:
                st.markdown("""
                    <div style="text-align: center; font-size: 14px; font-weight: bold; color: grey;">
                        No Opportunity found today, bored... 😴
                    </div>
                """, unsafe_allow_html=True)
            else:
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

            results = find_alert_symbols(cur_alert_dict, "main_accumulating")
            st.markdown("""
                <div style="text-align: center; font-size: 18px; font-weight: bold; color: #FFA500;">
                    Uptrend Position Building
                </div>
            """, unsafe_allow_html=True)
            if not results:
                st.markdown("""
                    <div style="text-align: center; font-size: 14px; font-weight: bold; color: grey;">
                        No Opportunity found today, bored... 😴
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <style>
                        .sell-container {
                            display: flex;
                            flex-wrap: wrap;
                            gap: 10px;
                        }
                        .sell-badge {
                            background-color: #FFA500 !important;
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

        # Mid Term Column
        with col2:
            results = find_alert_symbols(cur_alert_dict, 'long_accelerating')
            st.markdown("""
                <div style="text-align: center; font-size: 18px; font-weight: bold; color: #4CAF50;">
                    Accelerating 
                </div>
            """, unsafe_allow_html=True)
            if not results:
                st.markdown("""
                    <div style="text-align: center; font-size: 14px; font-weight: bold; color: grey;">
                        No Opportunity found today, bored... 😴
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="buy-container">', unsafe_allow_html=True)
                for symbol in results:
                    st.markdown(f'<div class="buy-badge">{symbol}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            results = find_alert_symbols(cur_alert_dict, "long_main_accumulating")
            st.markdown("""
                <div style="text-align: center; font-size: 18px; font-weight: bold; color: #FFA500;">
                    Uptrend Position Building
                </div>
            """, unsafe_allow_html=True)
            if not results:
                st.markdown("""
                    <div style="text-align: center; font-size: 14px; font-weight: bold; color: grey;">
                        No Opportunity found today, bored... 😴
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="sell-container">', unsafe_allow_html=True)
                for symbol in results:
                    st.markdown(f'<div class="sell-badge">{symbol}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # Long Term Column
        with col3:
            results = find_alert_symbols(cur_alert_dict, 'ext_long_accelerating')
            st.markdown("""
                <div style="text-align: center; font-size: 18px; font-weight: bold; color: #4CAF50;">
                    Accelerating 
                </div>
            """, unsafe_allow_html=True)
            if not results:
                st.markdown("""
                    <div style="text-align: center; font-size: 14px; font-weight: bold; color: grey;">
                        No Opportunity found today, bored... 😴
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="buy-container">', unsafe_allow_html=True)
                for symbol in results:
                    st.markdown(f'<div class="buy-badge">{symbol}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            results = find_alert_symbols(cur_alert_dict, "ext_accumulating")
            st.markdown("""
                <div style="text-align: center; font-size: 18px; font-weight: bold; color: #FFA500;">
                    Uptrend Position Building
                </div>
            """, unsafe_allow_html=True)
            if not results:
                st.markdown("""
                    <div style="text-align: center; font-size: 14px; font-weight: bold; color: grey;">
                        No Opportunity found today, bored... 😴
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="sell-container">', unsafe_allow_html=True)
                for symbol in results:
                    st.markdown(f'<div class="sell-badge">{symbol}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

def user_dashboard():
    # Welcom message
    username = (st.session_state['username']).capitalize()
    # Scrolling message of sandbox testing results
    sandbox_results = initialize_mongo_client()[DB_NAME][SANDBOX_COLLECTION]
    sandbox_results_list = list(sandbox_results.find({}))
    scrolling_message = analyze_strategy_results(sandbox_results_list)
    
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
                    padding-left: 1%; /* Start outside view */
                    font-size: 18px;
                    font-weight: bold;
                    animation: scroll 45s linear infinite; 
                }

                /* Define the scrolling animation */
                @keyframes scroll {
                    100% { transform: translateX(0%); } /* Start closer */
                    100% { transform: translateX(-100%); }
                }
                </style>
        """, unsafe_allow_html=True)
    # Create scrolling text with metrics with colored values
    scrolling_text = (
        f'Win Rate: <span style="color: {"#4CAF50" if scrolling_message["win_rate"] > 50 else "#FF5733"}">{scrolling_message["win_rate"]}%</span> | '
        f'Average Profit: <span style="color: #4CAF50">{scrolling_message["avg_profit"]}%</span> | '
        f'Average Loss: <span style="color: #FF5733">{scrolling_message["avg_loss"]}%</span> | '
        f'Maximum Drawdown: <span style="color: {"#FF5733" if scrolling_message["max_drawdown"] > 20 else "#FFA500"}">{scrolling_message["max_drawdown"]}%</span>'
    )
    # Display scrolling marquee in Streamlit
    st.markdown(
        f"""
        <div class="marquee-container">
            <span class="marquee">
                CondVest 2024 Long Term Opportunity Alert: {scrolling_text}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(f"<h1 style='text-align: center;'>{username}'s Dashboard </h1>", unsafe_allow_html=True)

    # Prepare data for display data
    most_recent_trade_date = pd.to_datetime(get_most_current_trading_date())
    candidate_collection = initialize_mongo_client()[DB_NAME][CANDI_COLLECTION]
    current_alerts_dict = list(candidate_collection.find({"date": {"$gte": most_recent_trade_date}}))

    # Display all content
    display_user_dashboard_content(cur_alert_dict=current_alerts_dict)

def main():
    user_dashboard()

