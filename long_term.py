import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import plotly.subplots as sp
import io, redis, time, json, pymongo
from streamlit_autorefresh import st_autorefresh

# MongoDB Configuration
DB_NAME = st.secrets['mongo']['db_name']
PROCESSED_COLLECTION_NAME = st.secrets['mongo']['processed_collection_name']
ALERT_COLLECTION_NAME = st.secrets['mongo']['alert_collection_name']

# Redis Configuration
REDIS_HOST = st.secrets['redis']['host']
REDIS_PORT = st.secrets['redis']['port']
REDIS_PASSWORD = st.secrets['redis']['password']

def initialize_redis():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD)

def connect_to_mongo():
    client = pymongo.MongoClient(st.secrets["mongo"]["host"])
    return client[DB_NAME]

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

def fetch_alert_data(redis_client, collection, stock_symbol, interval):
    start_time = time.time()
    redis_key = f"alert_data:{stock_symbol}:{interval}"
    
    # Try to get cached data from Redis
    try:
        cached_data = redis_client.get(redis_key)
        if cached_data:
            return pd.read_json(io.StringIO(cached_data.decode("utf-8")))
    except Exception as e:
        st.warning(f"Redis error: {str(e)}")
    
    # Fetch and process MongoDB data using aggregation pipeline
    pipeline = [
        {"$match": {"symbol": stock_symbol, "interval": interval, "instrument": "equity"}},
        {"$project": {
            "_id": 0,
            "date": 1,
            "symbol": 1,
            "interval": 1,
            "instrument": 1,
            "open": 1, 
            "high": 1,
            "low": 1,
            "close": 1,
            "volume": 1,
            "momentum_alert": {"$ifNull": [{"$getField": {"field": "alert_type", "input": {"$getField": {"field": "momentum_alert", "input": "$alerts"}}}}, None]},
            "velocity_alert": {"$ifNull": [{"$getField": {"field": "alert_type", "input": {"$getField": {"field": "velocity_alert", "input": "$alerts"}}}}, None]},
            "touch_type": {
                "$cond": {
                    "if": {"$ifNull": [{"$getField": {"field": "169ema_touched", "input": "$alerts"}}, False]},
                    "then": {"$getField": {"field": "type", "input": {"$getField": {"field": "169ema_touched", "input": "$alerts"}}}},
                    "else": {
                        "$cond": {
                            "if": {"$ifNull": [{"$getField": {"field": "13ema_touched", "input": "$alerts"}}, False]},
                            "then": {"$getField": {"field": "type", "input": {"$getField": {"field": "13ema_touched", "input": "$alerts"}}}},
                            "else": None
                        }
                    }
                }
            }
        }}
    ]
    
    data = list(collection.aggregate(pipeline))
    df = pd.DataFrame(data).sort_values(by=['date'])
    
    # Cache in Redis
    try:
        redis_client.setex(redis_key, 120, df.to_json(orient="records"))
    except Exception as e:
        st.warning(f"Redis caching error: {str(e)}")
            
    return df

def fetch_latest_stock_data(redis_client, symbol):
    """
    Fetches the latest stock data from Redis cache for a given symbol.
    """
    try:
        realtime_key = f"live_trade:{symbol}"
        cached_data = redis_client.hgetall(realtime_key)
        if cached_data:
            decoded_data = {key.decode(): float(value.decode()) for key, value in cached_data.items()}
            return decoded_data
        return None
    except Exception as e:
        st.warning(f"Error fetching latest data for {symbol}: {str(e)}")
        return None
    

def candle_chart(filtered_df, latest_data):

    # If live data exists, append it to the dataframe
    if latest_data and all(v is not None for v in latest_data.values()) and filtered_df['interval'].iloc[-1] == 1:
        # Create new row with latest data
        latest_data = pd.DataFrame([latest_data])
        
        # Inherit all columns from filtered_df
        for col in filtered_df.columns:
            if col not in latest_data.columns:
                latest_data[col] = filtered_df[col].iloc[-1]
        
        # Increment date by 1 day
        latest_data['date'] = filtered_df['date'].max() + pd.Timedelta(days=1)
        
        # Concatenate with original dataframe
        filtered_df = pd.concat([filtered_df, latest_data])
        # Calculate date range and price range for last 6 months
    end_date = filtered_df['date'].max() + pd.DateOffset(days=20)
    start_date = end_date - pd.DateOffset(months=6)
    
    price_range = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]
    min_price = price_range['low'].min() * 0.95
    max_price = price_range['high'].max() * 1.05

    # Create figure with candlestick chart
    fig = sp.make_subplots(rows=1, cols=1, shared_xaxes=True)
    
    fig.add_trace(go.Candlestick(
        x=filtered_df['date'],
        open=filtered_df['open'], 
        high=filtered_df['high'],
        low=filtered_df['low'],
        close=filtered_df['close'],
        name='price'))

    # Add EMA lines
    for ema in ['144ema', '169ema', '13ema', '8ema']:
        fig.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df[ema], mode="lines", name=ema))

    # Update axes and layout
    fig.update_xaxes(range=[start_date, end_date], title_text="Date")
    fig.update_yaxes(range=[min_price, max_price], title_text="Price")
    fig.update_layout(xaxis_rangeslider_visible=False, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))

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

def static_analysis_page(processed_col, alert_col, redis_client):
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
            interval_mapping = {
                                1: "Short Term",
                                3: "Short-Medium Term",
                                5: "Medium Term",
                                8: "Medium-Long Term",
                                13: "Long Term"}

            distinct_intervals = alert_col.distinct("interval")
            interval_labels = [interval_mapping[interval] for interval in distinct_intervals]
            label_to_interval = {v: k for k, v in interval_mapping.items()}

            # Slidebar for interval selection
            selected_label = st.select_slider(" ", options=interval_labels, key='interval_slider')
            interval_selector = label_to_interval[selected_label]
            
    chart_container = st.container()
    with chart_container:
        with st.spinner("Loading data..."):
            processed_df = fetch_stock_data(redis_client, processed_col, stock_selector, interval_selector)
            latest_data = fetch_latest_stock_data(redis_client, stock_selector)

        candlesticks_chart, fundmentals_chart = st.columns([3, 1])
        with candlesticks_chart:
            # Add auto-refresh functionality
            st_autorefresh(interval=600000, key="chart")

            # Create placeholder for the chart
            chart_placeholder = st.empty()
            fig = candle_chart(processed_df, latest_data)
            chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"chart_{time.time()}")
            
        with fundmentals_chart:
            fundemental_chart(processed_df)

    # Display the alerts
    with st.spinner("Loading data..."):
        alert_df = fetch_alert_data(redis_client, alert_col, stock_selector, interval_selector)
        
    display_alerts(alert_df)

def long_term_dashboard():
    redis_client = initialize_redis()
    # Connect to MongoDB and fetch the processed collection
    processed_collection = connect_to_mongo()[PROCESSED_COLLECTION_NAME]
    alert_collection = connect_to_mongo()[ALERT_COLLECTION_NAME]
    static_analysis_page(processed_collection, alert_collection, redis_client)
