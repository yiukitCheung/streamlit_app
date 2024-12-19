import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import plotly.subplots as sp
import io, redis, time, json, pymongo, yfinance as yf
from streamlit_autorefresh import st_autorefresh
from dependencies import search_stock, add_stock_to_database, check_symbol_yahoo
import gettext
import os
# MongoDB Configuration
DB_NAME = st.secrets['mongo']['db_name']
PROCESSED_COLLECTION_NAME = st.secrets['mongo']['processed_collection_name']
ALERT_COLLECTION_NAME = st.secrets['mongo']['alert_collection_name']

# Redis Configuration
REDIS_HOST = st.secrets['redis']['host']
REDIS_PORT = st.secrets['redis']['port']
REDIS_PASSWORD = st.secrets['redis']['password']

# ============================
# Configuration Section
# ============================
def initialize_redis():
    try:
        return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD)
    except Exception as e:
        st.error(f"Error initializing Redis client: {e}")
        return None

def connect_to_mongo():
    try:
        client = pymongo.MongoClient(st.secrets["mongo"]["host"])
        return client[DB_NAME]
    except Exception as e:
        st.error(f"Error initializing MongoDB client: {e}")
        return None

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
            },
            "fibonacci_retracement": {"$ifNull": [{"$getField": {"field": "fibonacci_retracement", "input": "$structural_area"}}, None]},
            "kernel_density_estimation": {"$ifNull": [{"$getField": {"field": "kernel_density_estimation", "input": "$structural_area"}}, None]}
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

def compute_price_change(latest_price, previous_price):
    if previous_price is None:
        return None
    return ((latest_price - previous_price) / previous_price) * 100

# ============================
# Chart Section
# ============================
def candle_chart(filtered_df, latest_data, alert_df):

    # Get the structural areas from alert_df directly
    # Get fibonacci levels
    fib_236 = alert_df['fibonacci_retracement'].iloc[-1]['fib_236']
    fib_382 = alert_df['fibonacci_retracement'].iloc[-1]['fib_382']
    fib_618 = alert_df['fibonacci_retracement'].iloc[-1]['fib_618'] 
    fib_786 = alert_df['fibonacci_retracement'].iloc[-1]['fib_786']
    fib_1236 = alert_df['fibonacci_retracement'].iloc[-1]['fib_1236']
    fib_1382 = alert_df['fibonacci_retracement'].iloc[-1]['fib_1382']

    # Get kernel density levels
    top = alert_df['kernel_density_estimation'].iloc[-1]['top']
    bottom = alert_df['kernel_density_estimation'].iloc[-1]['bottom']
    second_top = alert_df['kernel_density_estimation'].iloc[-1]['second_top']
    second_bottom = alert_df['kernel_density_estimation'].iloc[-1]['second_bottom']
    
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
    
    # Add candlestick trace
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
        
    # Add horizontal lines for fibonacci retracement levels
    fib_levels = [
        (fib_786, '78.6%', 'rgba(255,0,0,0.2)'),
        (fib_618, '61.8%', 'rgba(0,0,255,0.2)'),
        (fib_382, '38.2%', 'rgba(255,165,0,0.2)'),
        (fib_236, '23.6%', 'rgba(0,255,0,0.2)'),
        (fib_1236, '12.36%', 'rgba(0,0,255,0.2)'),
        (fib_1382, '13.82%', 'rgba(0,0,255,0.2)')
    ]

    if st.session_state.show_fibonacci:
        for level, pct, color in fib_levels:
            fig.add_hline(
                y=level,
                line_dash="solid",
                line_color=color,
                annotation_text=f'{pct} ({level:.2f})',
                annotation_font_size=12
            )

    # Add Dense Trading Range
    trading_ranges = [
        # Main Trading Area
        (top, bottom, 'rgba(0,0,255,0.1)', 'rgba(255,0,0,0.1)'),
        # Second structural area  
        (second_top, second_bottom, 'rgba(255,165,0,0.1)', 'rgba(255,165,0,0.1)')
    ]

    if st.session_state.show_trading_areas:
        for top_level, bottom_level, top_color, fill_color in trading_ranges:
            fig.add_traces([
                go.Scatter(
                    x=filtered_df['date'],
                    y=[top_level] * len(filtered_df),
                    fill=None,
                    mode='lines', 
                    line=dict(color=top_color),
                    name='Top'
                ),
                go.Scatter(
                    x=filtered_df['date'], 
                    y=[bottom_level] * len(filtered_df),
                    fill='tonexty',
                    mode='lines',
                    line=dict(color=fill_color),
                    fillcolor=fill_color,
                    name='Bottom'
                )
            ])
    # Update axes and layout
    fig.update_xaxes(range=[start_date, end_date], title_text="Date")
    fig.update_yaxes(range=[min_price, max_price], title_text="Price")
    
    # Update layout
    fig.update_layout(
        xaxis_rangeslider_visible=False, 
        showlegend=False, 
        margin=dict(t=30, b=10, l=10, r=10),
        title=dict(
            text=filtered_df['symbol'].iloc[0],
            x=0.5,
            y=0.95
        )
    )

    return fig

def display_alerts(alert_df):
    # Display Alert
    today_alert = alert_df[alert_df['date'] == alert_df['date'].max()]
    # Create columns for the three alert types
    col1, col2, col3 = st.columns(3)
    
    # Function to map alert values to color and message
    def get_alert_color_and_message(alert_type, value):
        alert_mappings = {
            'velocity_alert': {
                'velocity_maintained': (_("Maintained"), 'green'),
                'velocity_weak': (_("Weakened"), 'orange'), 
                'velocity_loss': (_("Loss"), 'red'),
                'velocity_negotiating': (_("Negotiating"), 'orange')
            },
            'touch_type': {
                'support': (_("Support"), 'green'),
                'resistance': (_("Resistance"), 'red'),
                
            },
            'momentum_alert': {
                "accelerated": (_("Accelerating"), 'green'),
                'decelerated': (_("Decelerating"), 'red'),
                
            }
        }
        
        # Default to grey color if value or alert type is not recognized
        message, color = alert_mappings.get(alert_type, {}).get(value, (_("No Alert"), 'grey'))
        return message, color

    # Display alerts in columns
    with col1:
        st.markdown("<h3 style='text-align: center;'>{}</h3>".format(
            _("Up Trend Strength")), 
            unsafe_allow_html=True)
        if 'velocity_alert' in today_alert.columns and not today_alert['velocity_alert'].empty:
            alert_value = today_alert['velocity_alert'].values[0]
            if pd.notna(alert_value):
                message, color = get_alert_color_and_message('velocity_alert', alert_value)
                st.markdown(f"""
                    <div style="text-align: center;">
                        <span style="font-size:50px; color:{color}">●</span>
                        <div style="font-size:16px; font-weight:bold; margin-top:10px; color:{color};">{message}</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h3 style='text-align: center;'>{}</h3>".format(
            _("Support/Resistance")), 
            unsafe_allow_html=True)
        if 'touch_type' in today_alert.columns:
            alert_value = today_alert['touch_type'].values[0]
            
            message, color = get_alert_color_and_message('touch_type', alert_value)

            st.markdown(f"""
                <div style="text-align: center;">
                    <span style="font-size:50px; color:{color}">●</span>
                    <div style="font-size:16px; font-weight:bold; margin-top:10px; color:{color};">{message}</div>
                </div>
                """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("<h3 style='text-align: center;'>{}</h3>".format(
            _("Momentum")), 
            unsafe_allow_html=True)
        if 'momentum_alert' in today_alert.columns:
            alert_value = today_alert['momentum_alert'].values[0]
            message, color = get_alert_color_and_message('momentum_alert', alert_value)
            st.markdown(f"""
                <div style="text-align: center;">
                    <span style="font-size:50px; color:{color}">●</span>
                    <div style="font-size:16px; font-weight:bold; margin-top:10px; color:{color};">{message}</div>
                </div>
                """, unsafe_allow_html=True)

def price_change_section(redis_client, stock_selector, processed_col):
    with st.container():
        # Fetch processed data
        processed_df = fetch_stock_data(redis_client, processed_col, stock_selector, 1)
        
        # Get latest data
        latest_data = fetch_latest_stock_data(redis_client, stock_selector) 
        
        # Dynamically compute price change
        # During trading hours
        if latest_data:
            
            # Get latest price
            latest_price = latest_data.get('close')
            previous_price = processed_df.iloc[-1]['close']
            
            # Compute price change
            price_change = compute_price_change(latest_price, previous_price)
            
            # Determine color and arrow
            color = 'green' if price_change > 0 else 'red'
            arrow = '▲' if price_change > 0 else '▼'
            
        # After trading hours
        elif not latest_data:
            # Get latest price
            latest_data = processed_df.iloc[-1]['close']
            previous_data = processed_df.iloc[-2]['close']
            
            # Compute price change
            price_change = compute_price_change(latest_data, previous_data)
            
            # Determine color and arrow
            color = 'green' if price_change > 0 else 'red'
            arrow = '▲' if price_change > 0 else '▼'
            
        # If no data is available
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

def price_section(redis_client, stock_selector, processed_col):
    with st.container():
        # Fetch processed data
        processed_df = fetch_stock_data(redis_client, processed_col, stock_selector, 1)
        
        # Get latest data
        latest_data = fetch_latest_stock_data(redis_client, stock_selector) 
        
        # Dynamically compute price change
        # During trading hours
        if latest_data:
            
            # Get latest price
            latest_price = latest_data.get('close')
            previous_price = processed_df.iloc[-1]['close']

            # Determine color and arrow
            color = 'green' if latest_price > previous_price else 'red'
            arrow = '▲' if latest_price > previous_price else '▼'
            
        # After trading hours
        elif not latest_data:
            # Get latest price
            latest_data = processed_df.iloc[-1]['close']
            previous_data = processed_df.iloc[-2]['close']
            
            # Determine color and arrow
            color = 'green' if latest_data > previous_data else 'red'
            arrow = '▲' if latest_data > previous_data else '▼'

        # If no data is available
        else:
            color = 'grey'
            arrow = ''
    
        # Display price change
        st.markdown(f"""
            <div style="text-align: center;">
                <div style="font-size:48px; font-weight:bold; color:{color};">
                    Price: {latest_price:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

def stock_config_section(alert_col):
    with st.container():
        # Reduce space around selection components
        st.markdown("""
            <style>
            .css-1y0tads {padding-top: 0px; padding-bottom: 0px;}
            </style>
            """, unsafe_allow_html=True)
        
        symbol_col, interval_col, show_structure_col = st.columns(3)
        with symbol_col:
            # Create a dropdown to select the stock
            stock_to_search = st.text_input("{}".format(
                _("Search Stock")), value="")
            if stock_to_search:
                try:
                    stock_selector = search_stock(stock_to_search)[0]
                except:
                    st.session_state.stock_not_found = True
                    st.warning("Sorry, Stock is not in database, would you like to add it?")
                    if st.button("Add Stock"):
                        if check_symbol_yahoo(stock_to_search.upper()):
                            stock_to_search = stock_to_search.upper()
                            full_name = yf.Ticker(stock_to_search).info.get('longName')
                            add_stock_to_database(stock_to_search, full_name)
                            st.success("🎉 Thanks for your contribution! Your stock will be added to the database tomorrow.")
                            st.balloons()
                        else:
                            st.warning("Sorry, this stock is not available on Market")
                    stock_selector = 'AAPL'
            else:
                stock_selector = 'AAPL'

        with interval_col:
            # Create a dropdown to select the interval
            interval_mapping = {
                                1: _("Short Term"),
                                3: _("Short-Medium Term"),
                                5: _("Medium Term"),
                                8: _("Medium-Long Term"),
                                13: _("Long Term")}

            distinct_intervals = alert_col.distinct("interval")
            interval_labels = [interval_mapping[interval] for interval in distinct_intervals]
            label_to_interval = {v: k for k, v in interval_mapping.items()}

            # Slidebar for interval selection
            selected_label = st.select_slider(" ", options=interval_labels, key='interval_slider')
            interval_selector = label_to_interval[selected_label]
            
        with show_structure_col:
            option_mapping = {
                'Key Price Points': _("Key Price Points"),
                'Most Trading Areas': _("Most Trading Areas")
            }
            # Display the options with default values
            selected_options = st.multiselect("{}".format(
                _("Key Price Areas")), options=list(option_mapping.values()), default=list(option_mapping.values()))
            # Set defaults
            st.session_state.show_structure = False
            st.session_state.show_fibonacci = _("Key Price Points") in selected_options
            st.session_state.show_trading_areas = _("Most Trading Areas") in selected_options
        
        return stock_selector, interval_selector

def chart_section(processed_col, alert_col, redis_client, stock_selector, interval_selector):
    with st.container():
        with st.spinner("{}".format(
            _("Loading data..."))):
            processed_df = fetch_stock_data(redis_client, processed_col, stock_selector, interval_selector)
            latest_data = fetch_latest_stock_data(redis_client, stock_selector)

        with st.spinner("{}".format(
            _("Loading Alert Data..."))):
            alert_df = fetch_alert_data(redis_client, alert_col, stock_selector, interval_selector)
        # Create placeholder for the chart
        fig = candle_chart(processed_df, latest_data, alert_df)
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{time.time()}")
            
    return alert_df

def Stock_Indepth_Dashboard(processed_col, alert_col, redis_client):
    # Initialize session state
    for state in ['show_fibonacci', 'show_trading_areas', 'stock_not_found']:
        if state not in st.session_state:
            st.session_state[state] = False
            
    st.markdown("<h2 style='text-align: center;'>{}</h2>".format(
        _("Stock Indepth Dashboard")), 
        unsafe_allow_html=True)
    
    # Stock Config Section
    stock_selector, interval_selector = stock_config_section(alert_col)
    
    chart_col, price_col = st.columns([4, 1])
    with chart_col:
        # Chart Section
        alert_df = chart_section(processed_col, alert_col, redis_client, stock_selector, interval_selector)
    
    with price_col:
        # Price Section
        price_section(redis_client, stock_selector, processed_col)
        
        # Price Change Section
        price_change_section(redis_client, stock_selector, processed_col)
        
    # Display the alerts
    display_alerts(alert_df)

def long_term_dashboard():
    # Set up translations
    locale_dir = os.path.join(os.path.dirname(__file__), 'locale')
    translation = gettext.translation(
        'messages',
        localedir=locale_dir, 
        languages=['en'],
        fallback=True
    )
    translation.install()
    global _
    _ = translation.gettext

    # Initialize necessary connections
    redis_client = initialize_redis()
    db = connect_to_mongo()
    
    # Display the dashboard
    Stock_Indepth_Dashboard(
        db[PROCESSED_COLLECTION_NAME],
        db[ALERT_COLLECTION_NAME], 
        redis_client
    )
