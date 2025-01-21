import pandas as pd
import pymongo
import plotly.graph_objects as go
import streamlit as st
import time
from streamlit_autorefresh import st_autorefresh
import gettext
import os
import io
import redis
import numpy as np
# MongoDB Configuration
DB_NAME = st.secrets['mongo']['db_name']
PROCESSED_COLLECTION_NAME = st.secrets['mongo']['processed_collection_name']
STREAMING_COLLECTIONS = [f'{interval}_datastream' for interval in st.secrets['mongo']['streaming_interval']]
LIVE_ALERT_COLLECTION = st.secrets['mongo']['live']
SUP_RES_BATCH_ALERT_COLLECTION = st.secrets['mongo']['sup_res_batch']
VOLUME_SPIKE_BATCH_ALERT_COLLECTION = st.secrets['mongo']['volume_spike_batch']
DESIRED_STREAMING_INTERVAL = st.secrets['mongo']['streaming_interval']
WINDOW_SIZE = st.secrets['mongo']['window_size']
REDIS_HOST = st.secrets['redis']['host']
REDIS_PORT = st.secrets['redis']['port']
REDIS_PASSWORD = st.secrets['redis']['password']

def initialize_redis():
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD)
    return redis_client

@st.cache_data
def get_current_date():
    current_date = pd.to_datetime('today').normalize()
    if current_date.weekday() in [5, 6]:  # Saturday or Sunday
        current_date -= pd.Timedelta(days=current_date.weekday() - 4)
    elif current_date.weekday() == 0 and pd.to_datetime('today').hour < 9:  # Monday before 9:30
        current_date -= pd.Timedelta(days=3)
    elif pd.to_datetime('today').hour < 7:  # Weekday before 9:30
        current_date -= pd.Timedelta(days=1)
    return current_date.replace(hour=7, minute=30) - pd.Timedelta(days=WINDOW_SIZE)


def fetch_data(db, collection_name, symbol):
    try:
        query = {"symbol": symbol}
        return db[collection_name].find(query)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()


def fetch_live_alerts(db, collection_name, symbol, interval, date, alert_type=None):
    try:
        query = {"symbol": symbol, "interval": interval, "datetime": {"$gte": date}}
        if alert_type:
            query["alert_type"] = {"$in": alert_type}

        # Fetch the latest document by sorting 'datetime' in descending order
        return db[collection_name].find(query, sort=[("datetime", -1)])
    except Exception as e:
        st.error(f"Error fetching live alerts: {e}")
        return pd.DataFrame()

def fetch_batch_alerts(db, collection_name, symbol, interval):
    try:
        query = {"symbol": symbol, "interval": interval}
        return db[collection_name].find(query)
    except Exception as e:
        st.error(f"Error fetching batch alerts: {e}")
        return pd.DataFrame()

def fetch_stock_data(redis_client, collection, stock_symbol, interval):
    start_time = time.time()
    warehouse_interval = st.secrets['mongo']['warehouse_interval']
    if not warehouse_interval:
        raise ValueError("warehouse_interval is empty in st.secrets")

    # Define Redis key based on stock symbol and interval
    redis_key = f"stock_data:{stock_symbol}:{interval}"

    try:
        # Check if data is cached in Redis and deserialize in one step
        if cached_data := redis_client.get(redis_key):
            data = pd.read_json(io.StringIO(cached_data.decode("utf-8")))
        else:
            # Use MongoDB projection and sorting at database level
            cursor = collection.find(
                {
                    "symbol": stock_symbol,
                    "interval": interval, 
                    "instrument": "equity",
                    "date": {"$gte": pd.Timestamp.now() - pd.Timedelta(days=1825)}
                },
                {"_id": 0}
            ).sort("date", 1)
            
            # Create DataFrame directly from cursor
            data = pd.DataFrame(cursor)
            
            # Cache the result with compression
            redis_client.setex(
                redis_key,
                300,
                data.to_json(orient="records", date_format='iso')
            )

        if st.session_state.get('debug'):
            end_time = time.time()
            st.write(f"Time taken to fetch data: {end_time - start_time:.2f} seconds")
            
        return data

    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error

def fetch_latest_stock_data(redis_client, symbol, interval, db):
    """
    Fetches the latest stock data from Redis cache for a given symbol.
    """
    try:
        realtime_key = f"live_trade:{symbol}"
        cached_data = redis_client.hgetall(realtime_key)
        if cached_data:
            decoded_data = {key.decode(): float(value.decode()) for key, value in cached_data.items()}
            return decoded_data
        else:
            # Use MongoDB to fetch the latest data
            query = {"symbol": symbol}
            price = db[STREAMING_COLLECTIONS[0]].find_one(query, sort=[("datetime", -1)])
            return price
        
    except Exception as e:
        st.warning(f"Error fetching latest data for {symbol}: {str(e)}")
        return None
    
def plot_candlestick_chart(filtered_df, support_resistance_alert_df, vol_spike_alert_df):
    # Extract the interval in minutes for gap filling
    interval_in_minutes = int(filtered_df['interval'].iloc[0].replace('m',''))\
                            if filtered_df['interval'].iloc[0].endswith('m')\
                                else int(filtered_df['interval'].iloc[0].replace('h','')) * 60
    
    # Create figure for candlestick chart
    fig = go.Figure()
    # Add candlestick chart
    fig.add_trace(go.Candlestick(x=filtered_df['datetime'],
                                open=filtered_df['open'],
                                high=filtered_df['high'],
                                low=filtered_df['low'],
                                close=filtered_df['close'],
                                increasing_line_color='green',
                                decreasing_line_color='red'
                            ))
    
    # Add support and resistance areas to the candlestick chart
    
    if len(support_resistance_alert_df) > 0:
        # Get the latest support and resistance values
        latest_ressistance = support_resistance_alert_df[pd.notna(support_resistance_alert_df['resistance_upper'])].sort_values('datetime', ascending=False).iloc[0][['resistance_upper', 'resistance_lower']]
        latest_support = support_resistance_alert_df[pd.notna(support_resistance_alert_df['support_upper'])].sort_values('datetime', ascending=False).iloc[0][['support_upper', 'support_lower']]
        
        # Plot latest support area if exists
        if not pd.isna(latest_support.get('support_upper')) and not pd.isna(latest_support.get('support_lower')):
            fig.add_trace(go.Scatter(
                x=[filtered_df['datetime'].min(), filtered_df['datetime'].max()],
                y=[latest_support['support_upper'], latest_support['support_upper']],
                fill=None,
                mode='lines',
                line=dict(color='rgba(0,0,255,0.3)', width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[filtered_df['datetime'].min(), filtered_df['datetime'].max()],
                y=[latest_support['support_lower'], latest_support['support_lower']],
                fill='tonexty',
                mode='lines',
                line=dict(color='rgba(0,0,255,0.3)', width=0),
                fillcolor='rgba(0,0,255,0.3)',
                showlegend=False
            ))

        # Plot latest resistance area if exists
        if not pd.isna(latest_ressistance.get('resistance_upper')) and not pd.isna(latest_ressistance.get('resistance_lower')):
            fig.add_trace(go.Scatter(
                x=[filtered_df['datetime'].min(), filtered_df['datetime'].max()],
                y=[latest_ressistance['resistance_upper'], latest_ressistance['resistance_upper']],
                fill=None,
                mode='lines',
                line=dict(color='rgba(255,0,0,0.3)', width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[filtered_df['datetime'].min(), filtered_df['datetime'].max()],
                y=[latest_ressistance['resistance_lower'], latest_ressistance['resistance_lower']],
                fill='tonexty',
                mode='lines',
                line=dict(color='rgba(255,0,0,0.3)', width=0),
                fillcolor='rgba(255,0,0,0.3)',
                showlegend=False
            ))
    # Add latest volume spikes to the candlestick chart
    if len(vol_spike_alert_df) > 0:
        # Plot the volume spike
        latest_vol_spike = vol_spike_alert_df.sort_values('datetime').iloc[-1]
        if latest_vol_spike.get('bullish_volume_spike') == 1.0 and pd.notna(latest_vol_spike['datetime']):
            fig.add_trace(go.Scatter(x=[latest_vol_spike['datetime']],
                                    y=[latest_vol_spike['close'] * 1.005], 
                                    mode='markers', 
                                    marker=dict(symbol='arrow-up', color='green', size=24), 
                                    name='Bullish'))
        elif latest_vol_spike.get('bearish_volume_spike') == 1.0 and pd.notna(latest_vol_spike['datetime']):
            fig.add_trace(go.Scatter(x=[latest_vol_spike['datetime']], 
                                    y=[latest_vol_spike['close'] * 1.005], 
                                    mode='markers', 
                                    marker=dict(symbol='arrow-down', color='red', size=24), 
                                    name='Bearish'))   
                
    # grab first and last observations from df.date and make a continuous date range from that
    dt_all = pd.date_range(start=filtered_df['datetime'].iloc[0],end=filtered_df['datetime'].iloc[-1], freq = f'{interval_in_minutes}min')
    
    # check which dates from your source that also accur in the continuous date range
    dt_obs = [d.strftime("%Y-%m-%d %H:%M:%S") for d in filtered_df['datetime']]

    # isolate missing timestamps
    dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d %H:%M:%S").tolist() if not d in dt_obs]
    dt_breaks = pd.to_datetime(dt_breaks)
    
    # Update x-axis
    days_to_display = 1 if interval_in_minutes == 5 else 4
    max_display_date = filtered_df['datetime'].max() + pd.Timedelta(hours=1)
    min_display_date = max_display_date - pd.Timedelta(days=days_to_display)

    # Update y-axis
    price_range = filtered_df[(filtered_df['datetime'] >= min_display_date) & (filtered_df['datetime'] <= max_display_date)]    
    min_price = price_range['low'].min() * 0.99
    max_price = price_range['high'].max() * 1.01
    
    # Update x-axis
    fig.update_xaxes(
        rangebreaks=[dict(dvalue=interval_in_minutes * 60 * 1000, values=dt_breaks)],
        range=[min_display_date, max_display_date]
    )
    
    fig.update_yaxes(range=[min_price, max_price])
    
    # Update layout
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        showlegend=False,
        height=600
    )
    
    return fig

def compute_price_change(latest_price, previous_price):
    if previous_price is None:
        return None
    return ((latest_price - previous_price) / previous_price) * 100

def chart_section(db, stock_selector, interval):
    # Prepare data for the selected stock and interval
    selected_topic = f"{interval}_datastream"
    filtered_query = fetch_data(db, selected_topic, stock_selector)
    filtered_live_alerts_query = fetch_live_alerts(db, LIVE_ALERT_COLLECTION, 
                                            stock_selector,   
                                            interval,
                                            date=get_current_date(),
                                            alert_type=["bullish_engulfer", "bearish_engulfer", "bullish_382"])
    # Fetch the batch alerts
    filtered_sup_res_batch_alerts_query = fetch_batch_alerts(db, SUP_RES_BATCH_ALERT_COLLECTION, 
                                            stock_selector, 
                                            interval)
    filtered_volume_spike_batch_alerts_query = fetch_batch_alerts(db, VOLUME_SPIKE_BATCH_ALERT_COLLECTION, 
                                                                stock_selector, 
                                                                interval)
    # Convert the query results to a DataFrame
    filtered_df = pd.DataFrame(list(filtered_query))
    filtered_live_alerts = pd.DataFrame(list(filtered_live_alerts_query))
    filtered_sup_res_batch_alerts = pd.DataFrame(list(filtered_sup_res_batch_alerts_query))
    filtered_volume_spike_batch_alerts = pd.DataFrame(list(filtered_volume_spike_batch_alerts_query))
    
    
    
    # Find the most frequent support
    if not filtered_df.empty:
        # Ensure consistency in the datetime format
        filtered_df['datetime'] = pd.to_datetime(filtered_df['datetime'])
    if not filtered_live_alerts.empty:
        filtered_live_alerts['datetime'] = pd.to_datetime(filtered_live_alerts['datetime'])

    # Plot the Chart
    candlestick_chart = plot_candlestick_chart(filtered_df, filtered_sup_res_batch_alerts, filtered_volume_spike_batch_alerts)
    
    # Display the chart
    st.plotly_chart(candlestick_chart, use_container_width=True, key=f"chart_{time.time()}")
    

def price_change_section(redis_client, stock_selector, processed_col, interval, db):
    # Fetch processed data
    processed_df = fetch_stock_data(redis_client, processed_col, stock_selector, 1)
    
    # Get latest data
    latest_data = fetch_latest_stock_data(redis_client, stock_selector, interval, db) 
    
    # Determine price change
    if latest_data:
        # Get latest price
        latest_price = latest_data.get('close')
        previous_price = processed_df.iloc[-1]['close']
    elif not latest_data and len(processed_df) > 1:
        # After trading hours, use the last two entries
        latest_price = processed_df.iloc[-1]['close']
        previous_price = processed_df.iloc[-2]['close']
    else:
        # If no data is available
        latest_price = None
        previous_price = None

    # Compute price change
    if latest_price is not None and previous_price is not None:
        price_change = compute_price_change(latest_price, previous_price)
        color = 'green' if price_change > 0 else 'red'
        arrow = '▲' if price_change > 0 else '▼'
    else:
        price_change = 0
        color = 'grey'
        arrow = ''

    # Display price change
    st.markdown(f"""
        <div style="text-align: center;">
            <div style="font-size:48px; font-weight:bold; color:{color};">
                {arrow} {'+' if price_change > 0 else ''}{price_change:.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

def price_section(redis_client, stock_selector, processed_col, interval, db):

    # Fetch processed data
    processed_df = fetch_stock_data(redis_client, processed_col, stock_selector, 1)
    
    # Get latest data
    latest_data = fetch_latest_stock_data(redis_client, stock_selector, interval, db) 
    

    # Dynamically compute price change
    # During trading hours
    if latest_data:
        
        # Get latest price
        latest_price = latest_data.get('close')
        previous_price = processed_df.iloc[-1]['close']

        # Determine color and arrow
        color = 'green' if latest_price > previous_price else 'red'
        
    # After trading hours
    elif not latest_data:
        # Get latest price
        latest_price = processed_df.iloc[-1]['close']
        previous_data = processed_df.iloc[-2]['close']
        # Determine color and arrow
        color = 'green' if latest_price > previous_data else 'red'

    # If no data is available
    else:
        color = 'grey'

    # Display price change
    st.markdown(f"""
        <div style="text-align: center;">
            <div style="font-size:48px; font-weight:bold; color:{color};">
                Price: {latest_price:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

def symbol_section(stock_selector):
    st.markdown(f"""
        <div style="text-align: center;">
            <div style="font-size:48px; font-weight:bold; color:black;">
                {stock_selector}
            </div>
        </div>
        """, unsafe_allow_html=True)

def options_section(db):
    with st.container():
        # Get the list of stocks
        options = sorted(db[STREAMING_COLLECTIONS[0]].find({'instrument':'equity'}).distinct("symbol"))
        intervals = DESIRED_STREAMING_INTERVAL        
        col1, col2 = st.columns([1, 1])
        with col1:
            stock_selector = st.selectbox(_("Select Stock"), options=options, index=options.index('AAPL') if 'AAPL' in options else 0, key="short_term_stock")
        with col2:
            intervals_selector = st.selectbox(_("Select Interval"), options=intervals, index=intervals.index('30m'), key="short_term_interval")
    
    return stock_selector, intervals_selector

def alert_section(db, redis_client, symbol, interval):
    st.markdown("""
        <style>
        .alert-container {
            background-color: var(--condvest-cream);
            border: 2px solid var(--condvest-gold);
            border-radius: 20px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 12px rgba(198, 169, 108, 0.15);
            height: 100%;
        }
        
        .zone-container {
            background-color: var(--condvest-beige);
            border: 1px solid var(--condvest-gold);
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            transition: all 0.3s ease;
            height: calc(100% - 20px);
        }
        
        .zone-header {
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 20px;
            letter-spacing: 0.8px;
            text-align: center;
            border-bottom: 2px solid var(--condvest-gold);
            padding-bottom: 12px;
            text-transform: uppercase;
        }
        
        .zone-value {
            font-size: 17px;
            color: var(--condvest-black);
            margin: 12px 0;
            padding: 10px 15px;
            background: rgba(255, 255, 255, 0.5);
            border-radius: 10px;
            display: block;
            text-align: center;
            font-weight: 500;
            letter-spacing: 0.5px;
        }
        
        .zone-change {
            font-size: 18px;
            font-weight: 600;
            margin-top: 15px;
            padding: 8px 15px;
            border-radius: 12px;
            display: block;
            text-align: center;
            letter-spacing: 0.5px;
        }

        .direction-arrow {
            font-size: 36px;
            font-weight: 700;
            margin-right: 5px;
        }
        
        .volume-alert {
            height: calc(100% - 20px);
            padding: 15px;
            background: var(--condvest-beige);
            border: 1px solid var(--condvest-gold);
            border-radius: 15px;
            margin-top: 0;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .alert-status {
            color: var(--condvest-black);
            text-align: center;
            padding: 15px;
            font-style: italic;
            font-size: 16px;
            font-weight: 500;
            background: rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            margin: 10px 0;
            letter-spacing: 0.5px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Fetch data efficiently
    try:
        # Fetch latest volume spike alert
        latest_vol_spike = db[VOLUME_SPIKE_BATCH_ALERT_COLLECTION].find_one(
            {"symbol": symbol, "interval": interval},
            sort=[("datetime", -1)]
        )
        
        # Fetch latest structural alerts with specific fields
        latest_resistance = db[SUP_RES_BATCH_ALERT_COLLECTION].find_one(
            {
                "symbol": symbol,
                "interval": interval,
                "resistance_upper": {"$ne": np.nan}
            },
            sort=[("datetime", -1)]
        )
        
        latest_support = db[SUP_RES_BATCH_ALERT_COLLECTION].find_one(
            {
                "symbol": symbol,
                "interval": interval, 
                "support_upper": {"$ne": np.nan}
            },
            sort=[("datetime", -1)]
        )

        # Get latest price
        latest_data = fetch_latest_stock_data(redis_client, symbol, interval, db)
        latest_price = latest_data.get('close') if latest_data else None
        
    except Exception as e:
        st.error(f"Error fetching alerts: {e}")
        return

    # Create three columns for the alerts
    col1, col2, col3 = st.columns(3)

    # Resistance Zone Column
    with col1:
        st.markdown("""
            <div class="zone-container">
                <div class="zone-header" style="color: var(--condvest-gold);">
                    RESISTANCE
                </div>
        """, unsafe_allow_html=True)

        if latest_resistance and latest_resistance.get('resistance_upper') is not None:
            res_high = latest_resistance.get('resistance_upper')
            res_low = latest_resistance.get('resistance_lower')
            
            if latest_price and res_high and res_low:
                is_above_resistance = latest_price > min(res_low, res_high)
                resistance_pct = ((latest_price - ((res_high + res_low)/2)) / ((res_high + res_low)/2) * 100)
                resistance_color = "#D64045" if is_above_resistance else "var(--condvest-black)"
                resistance_arrow = "▲" if resistance_pct > 0 else "▼"
                
                st.markdown(f"""
                    <div class="zone-value">Upper: {res_high:.2f}</div>
                    <div class="zone-value">Lower: {res_low:.2f}</div>
                    <div class="zone-change" style="color: {resistance_color}; background: rgba(214, 64, 69, 0.3);">
                        <span class="direction-arrow">{resistance_arrow}</span>{abs(resistance_pct):.1f}% from zone
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="alert-status">
                    No resistance zone detected
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Support Zone Column
    with col2:
        st.markdown("""
            <div class="zone-container">
                <div class="zone-header" style="color: var(--condvest-gold);">
                    SUPPORT
                </div>
        """, unsafe_allow_html=True)

        if latest_support and latest_support.get('support_upper') is not None:
            sup_high = latest_support.get('support_upper')
            sup_low = latest_support.get('support_lower')
            
            if latest_price and sup_high and sup_low:
                is_above_support = latest_price > min(sup_low, sup_high)
                support_pct = ((latest_price - ((sup_high + sup_low)/2)) / ((sup_high + sup_low)/2) * 100)
                support_color = "#2E8B57" if is_above_support else "var(--condvest-black)"
                support_arrow = "▲" if support_pct > 0 else "▼"
                
                st.markdown(f"""
                    <div class="zone-value">Upper: {sup_high:.2f}</div>
                    <div class="zone-value">Lower: {sup_low:.2f}</div>
                    <div class="zone-change" style="color: {support_color}; background: rgba(46, 139, 87, 0.3);">
                        <span class="direction-arrow">{support_arrow}</span>{abs(support_pct):.1f}% from zone
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="alert-status">
                    No support zone detected
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Volume Spike Alert Column
    with col3:
        st.markdown("""
            <div class="zone-container">
                <div class="zone-header" style="color: var(--condvest-gold);">
                    VOLUME ALERT
                </div>
        """, unsafe_allow_html=True)

        if latest_vol_spike and latest_price:
            alert_price = latest_vol_spike.get('close')
            if alert_price:
                pct_change = ((latest_price - alert_price) / alert_price) * 100
                color = "#2E8B57" if pct_change > 0 else "#D64045"
                arrow = "▲" if pct_change > 0 else "▼"
                arrow_color = "var(--condvest-black)" if pct_change > 0 else "var(--condvest-white)"
                st.markdown(f"""
                    <div class="zone-value">Alert Price: {alert_price:.2f}</div>
                    <div class="zone-change" style="color: {color}; background: {'rgba(46, 139, 87, 0.3)' if pct_change > 0 else 'rgba(214, 64, 69, 0.3)'}">
                        <span class="direction-arrow">{arrow}</span>{abs(pct_change):.1f}% since alert
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="alert-status">
                    No volume spike alerts
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

def short_term_dashboard():
    
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
        
    def initialize_mongo_client():
        client = pymongo.MongoClient(st.secrets["mongo"]["host"])
        return client[DB_NAME]

    # Initialize MongoDB client
    db = initialize_mongo_client()
    
    # Initialize Redis client
    redis_client = initialize_redis()
    
    # Get the options
    stock_selector, intervals_selector = options_section(db)
    
    # Get the processed collection
    processed_col = db[PROCESSED_COLLECTION_NAME]
    
    with st.container():
        
        # Show price and symbol
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:  
            symbol_section(stock_selector)
        with col2:
            price_section(redis_client, stock_selector, processed_col, intervals_selector, db)
        with col3:
            price_change_section(redis_client, stock_selector, processed_col, intervals_selector, db)

        # Continuously update the chart every minute
        chart_section(db, stock_selector, intervals_selector)
        
        # Show alerts
        alert_section(db, redis_client, stock_selector, intervals_selector)
