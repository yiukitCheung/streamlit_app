import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from pymongo import MongoClient
import yfinance as yf

from datetime import datetime
from long_term import long_term_dashboard
# Ensure the correct path to the 'data' directory
from stock_candidates_analysis import DailyTradingStrategy, Pick_Stock

# MongoDB Configuration
DB_NAME = st.secrets['db_name']
WAREHOUSE_INTERVAL = st.secrets.warehouse_interval
WAREHOUSE_INTERVAL_COLLECTION = '1d_data'
PROCESSED_COLLECTION = st.secrets.processed_collection_name
ALERT_COLLECTION = st.secrets.alert_collection_name
CANDI_COLLECTION = st.secrets.candidate_collection_name

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

# ======================================================================================== #
# Simulated Trading Plot comparison and Simulated Trading Statistics Results Presentation  #
# ======================================================================================== #

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
            ),
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
            x=filtered_df['date'],
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
                        dict(step="all", label="Max")  # Display all available data
                    ])
                ),
                rangeslider=dict(visible=False),  # Hide the range slider
                type="date"
            ),
            title={
                "text": f"{selected_symbols} Candlesticks",
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
                chart_type = st.selectbox("Chart Type", options=["Cumulative Return", "Candlesticks", "Line Chart"])
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
                selected_label = st.select_slider("Signal Intensity", options=interval_labels, key='interval_slider')
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
                    <div style="display: flex; flex-direction: column; align-items: flex-end; position: absolute; top: 20px; right: 20px; gap: 35px;">
                        <div style="display: flex; flex-direction: row; align-items: center; gap: 10px;">
                            <div style="width: 25px; height: 25px; background-color: {dot_colors[0]}; border-radius: 50%;"></div> 
                            <div style="width: 120px; font-size: 14px; color: #333; font-weight: bold; text-align: left;">Optimistic</div>
                        </div>
                        <div style="display: flex; flex-direction: row; align-items: center; gap: 10px;">
                            <div style="width: 25px; height: 25px; background-color: {dot_colors[4]}; border-radius: 50%;"></div> 
                            <div style="width: 120px; font-size: 14px; color: #333; font-weight: bold; text-align: left;">Neutral</div>
                        </div>
                        <div style="display: flex; flex-direction: row; align-items: center; gap: 10px;">
                            <div style="width: 25px; height: 25px; background-color: {dot_colors[1]}; border-radius: 50%;"></div>
                            <div style="width: 120px; font-size: 14px; color: #333; font-weight: bold; text-align: left;">Flag for Risk</div>
                        </div>
                        <div style="display: flex; flex-direction: row; align-items: center; gap: 10px;">
                            <div style="width: 25px; height: 25px; background-color: {dot_colors[2]}; border-radius: 50%;"></div>
                            <div style="width: 120px; font-size: 14px; color: #333; font-weight: bold; text-align: left;">Pessimistic</div>
                        </div>
                        <div style="display: flex; flex-direction: row; align-items: center; gap: 10px;">
                            <div style="width: 25px; height: 25px; background-color: {dot_colors[3]}; border-radius: 50%;"></div>
                            <div style="width: 120px; font-size: 14px; color: #333; font-weight: bold; text-align: left;">Consolidating</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                                
    # Opportunity of the day Section
    st.markdown(f"<h3 style='text-align: center;'>Opportunity of the Day </h3>", unsafe_allow_html=True)
    # Display the Accelerating and Uptrend Position Building Stocks
    left_comp, right_comp = st.columns(2)
    with left_comp:
        results = find_alert_symbols(cur_alert_dict, 'accelerating')
        st.markdown("""
            <div style="text-align: center; font-size: 24px; font-weight: bold; color: #4CAF50;">
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
    with right_comp:

        decelerated_results = find_alert_symbols(cur_alert_dict, "main_accumulating")
        results =  decelerated_results

        st.markdown("""
            <div style="text-align: center; font-size: 24px; font-weight: bold; color: #FFA500;">
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
            # Add Sell specific styles
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
    
def user_dashboard():
    # Welcom message
    username = (st.session_state['username']).capitalize()
    
    st.markdown(f"<h1 style='text-align: center;'>{username}'s Dashboard </h1>", unsafe_allow_html=True)

    # Prepare data for display data
    most_recent_trade_date = pd.to_datetime(get_most_current_trading_date())
    candidate_collection = initialize_mongo_client()[DB_NAME][CANDI_COLLECTION]
    current_alerts_dict = list(candidate_collection.find({"date": {"$gte": most_recent_trade_date}}))

    # Display all content
    display_user_dashboard_content(cur_alert_dict=current_alerts_dict)

def main():
    user_dashboard()

