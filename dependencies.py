import streamlit as st
import psycopg2
import os, re, hashlib
import pandas as pd
from pymongo import MongoClient
from twilio.rest import Client
import yfinance as yf
from openai import OpenAI
from polygon import RESTClient
import time
import io
import redis
import phonenumbers
import pycountry


# ChatGPT
chatgpt_client = OpenAI(api_key=st.secrets["chatgpt"]["api_key"])

# Database connection details
dbname = st.secrets["postgres"]["db_name_postgres"]
user = st.secrets["postgres"]["user"]
host = st.secrets['postgres']['host']
port = st.secrets['postgres']['port']
key = st.secrets['postgres']['password']

# Mongo
mongo_uri = st.secrets['mongo']['host']
db_name = st.secrets['mongo']['db_name']
portfolio_collection_name = st.secrets['mongo']['portfolio_collection_name']
sandbox_collection_name = st.secrets['mongo']['sandbox_collection_name']
processed_collection_name = st.secrets['mongo']['processed_collection_name']

# Polygon API
polygon_api_key = st.secrets['polygon']['api_key']

# Twilio
account_sid = st.secrets['twilio']['ACC_SID']
auth_token = st.secrets['twilio']['AUTH_TOKEN']

client = Client(account_sid, auth_token)
polygon_client = RESTClient(api_key=polygon_api_key)

# Function to get country codes
def get_country_codes():
    country_codes = []
    for country in pycountry.countries:
        try:
            dial_code = f"+{phonenumbers.country_code_for_region(country.alpha_2)}"
            country_codes.append(f"{country.name} ({dial_code})")
        except KeyError:
            # Skip countries without a dialing code
            continue
    return sorted(country_codes)

# Function to validate phone numbers
def validate_phone_number(phone, country_code):
    try:
        # Extract the country code (e.g., +1)
        region_code = country_code.split(" ")[-1].strip("()").strip("+")
        # Add the country code to the phone number
        phone = f"+{region_code}{phone}"
        parsed_number = phonenumbers.parse(phone, None)
        
        if phonenumbers.is_valid_number(parsed_number):
            return phone
        else:
            return None
    except phonenumbers.NumberParseException:
        return None
    
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

@st.cache_resource
def initialize_mongo_client():
    client = MongoClient(st.secrets["mongo"]["host"])
    return client

@st.cache_resource
def init_mongodb_portfolio():
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        if portfolio_collection_name not in db.list_collection_names():
            db.create_collection(portfolio_collection_name)
    except Exception as e:
        st.error(f"Error initializing MongoDB portfolio: {e}")
        return None
    
def init_postgres():
    try:
        conn = psycopg2.connect(database=dbname, user=user, host=host, port=port, password=key)
        cursor = conn.cursor()

        # Check if the users table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_schema='public' AND table_name='users'
            )
        """)
        table_exists = cursor.fetchone()[0]

        if not table_exists:
            cursor.execute('''CREATE TABLE public.users (
                                id SERIAL PRIMARY KEY,
                                username VARCHAR(225) UNIQUE NOT NULL,
                                password VARCHAR(225) NOT NULL,
                                mobile VARCHAR(225),
                                role VARCHAR(225) DEFAULT 'user')''')
            conn.commit()
            st.success("User Table Created!")

        # Check if admin account exists
        cursor.execute("SELECT username FROM public.users WHERE username = %s", ("admin",))
        admin_exists = cursor.fetchone()

        if not admin_exists:
            username, password, mobile, role = "admin", "password123", "1234567890", "admin"
            cursor.execute("INSERT INTO public.users (username, password, mobile, role) VALUES (%s, %s, %s, %s)",
                        (username, hash_password(password), mobile, role))
            conn.commit()
            st.success("Admin account created.")

        cursor.close()
        conn.close()


    except Exception as e:
        st.error(f"An error occurred: {e}")

def sign_up_process():
    twil_phone_number = st.secrets['twilio']['PHONE_NUMBER']
    with st.form(key='signup_form', clear_on_submit=True):
        st.header('Sign Up')
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        # Dropdown for country codes
        country_codes = get_country_codes()
        selected_country = st.selectbox("Select Your Country", country_codes)
        
        # Input for phone number
        phone_number = st.text_input("Phone Number")

        if st.form_submit_button("Create"):
            # Check if passwords match
            if password != confirm_password:
                st.error("Passwords do not match.")
                return  # Stop further processing

            # Validate username and phone number format
            if not validate_username(username):
                st.error("Invalid username format.")
                return
            # Validate phone number
            validated_phone = validate_phone_number(phone_number, selected_country)
            if not validated_phone:
                st.error("Please enter a valid phone number.")
                return

            # Hash the password and insert user data
            hashed_pw = hash_password(password)
            if insert_user_data(username, hashed_pw, validated_phone):
                st.success(f"Account created successfully for {username}!")
                send_welcome_msg(client, twil_phone_number, validated_phone, username, password)
                st.balloons()

                # Redirect to login page after successful signup
                st.session_state['show_login'] = True
                st.session_state['current_page'] = "Login"
            else:
                st.error("Error creating account. The username may already exist.")

def forgot_password():
    st.session_state["profile_verified"] = False
    # User verification form
    if not st.session_state.profile_verified:
        with st.form(key='verify_user_form'):
            username = st.text_input("What is your username?")
            phone_number = st.text_input("What is your phone number?")
            new_password = st.text_input("Enter your new password:", type="password")
            submit = st.form_submit_button("Reset Password")
        
        if submit:
            if username and phone_number:
                # Mocking database check
                if get_specific_username(username) and get_specific_phone_number(username, phone_number):
                    st.success("Profile found! Please reset your password.")
                    st.session_state.profile_verified = True
                    st.session_state.username = username
                    st.session_state.phone_number = phone_number
                    if reset_password(username, phone_number, new_password):
                        st.success("Password updated successfully!")
                    else:
                        st.error("Failed to update password.")
                else:
                    st.error("Invalid username or phone number.")
            else:
                st.error("Please provide both username and phone number.")

def get_usernames():
    try:
        conn = psycopg2.connect(database=dbname, password=key, user=user, host=host, port=port)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM public.users")
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        if result:
            return True
        else:
            return False
    except Exception as e:
        st.error(f"Error fetching usernames: {e}")
        return False

def get_specific_username(username):
    try:
        conn = psycopg2.connect(database=dbname, password=key, user=user, host=host, port=port)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM public.users WHERE username = %s", (username,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if result:
            return True
        else:
            return False
        
    except Exception as e:
        st.error(f"Error fetching usernames: {e}")
        
    return False

def get_phone_number():
    try:
        conn = psycopg2.connect(database=dbname, password=key, user=user, host=host, port=port)
        cursor = conn.cursor()
        cursor.execute("SELECT mobile FROM public.users")
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        return result
    except Exception as e:
        st.error(f"Error fetching phone numbers: {e}")
        return False

def get_specific_phone_number(username, phone_number):
    try:
        conn = psycopg2.connect(database=dbname, password=key, user=user, host=host, port=port)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM public.users WHERE username = %s AND mobile = %s", (username, phone_number))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        if result:
            return True
        else:
            return False
    except Exception as e:
        st.error(f"Error fetching phone numbers: {e}")

def validate_username(username):
    return bool(re.match(r'^[a-zA-Z0-9_]+$', username))

def insert_user_data(username, password, phone_number):
    try:
        conn = psycopg2.connect(database=dbname, password=key, user=user, host=host, port=port)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO public.users (username, password, mobile) VALUES (%s, %s, %s)",
                    (username, password, phone_number))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error inserting user data: {e}")
        return False

def send_welcome_msg(twilio_client, number, target_number, username, password):
    body = f"""
    Welcome to CondVest! 🎉

    Thank you for joining our community! We are passionately dedicated to ensuring your success in the financial markets. Our expert team works tirelessly to provide you with precise, timely buy/sell alerts and in-depth professional market analysis that will empower your investment decisions. We've designed our platform to be fully customizable - visit the settings page anytime to fine-tune your alert preferences and create a personalized trading experience that perfectly matches your unique investment style and goals. Your success is our mission, and we're here to support you every step of the way.

    Learn more about us:
    🌐 Website: https://condvest.streamlit.app
    💼 LinkedIn: https://ca.linkedin.com/company/condvest-inc?trk=public_post-text

    Our team of financial experts and data scientists are here to support your investment journey.

    Best regards,
    The CondVest Team
    
    Your Login Credentials:
    Username: {username}
    Password: {password}
    """

    try:
        twilio_client.messages.create(
            body=body,
            from_=number,
            to=target_number
        )
        st.info("Welcome message sent successfully!")
    except Exception as e:
        st.error(f"Failed to send welcome message: {e}")

def verify_user(username, password):
    try:
        # Replace with your actual PostgreSQL database credentials
        conn = psycopg2.connect(database=dbname, password=key, user=user, host=host, port=port)
        cursor = conn.cursor()
        cursor.execute("SELECT password, role FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if result:
            stored_hashed_password, role = result
            if stored_hashed_password == hash_password(password):
                return True, role
        return False, None
    except psycopg2.Error as e:
        st.error(f"Database error: {e}")
        return False, None

def fetch_symbol_portfolio(symbol: str, instrument: str):
    try:
        symbol = symbol.upper()
        instrument = instrument.lower()
        collection_obj = initialize_mongo_client()[db_name][processed_collection_name]
        cursor = collection_obj.find_one({'symbol': symbol, 'instrument': instrument}, projection={'symbol': True, '_id': False})
        
        if cursor:
            return pd.DataFrame(list(cursor))

        else:
            return None
    except Exception as e:
        return None

def reset_password(username, phone_number, new_password):
    try:
        conn = psycopg2.connect(database=dbname, password=key, user=user, host=host, port=port)
        cursor = conn.cursor()
        hashed_password = hash_password(new_password)
        cursor.execute("UPDATE public.users SET password = %s WHERE username = %s AND mobile = %s", (hashed_password, username, phone_number))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error resetting password: {e}")
        return False

def search_stock(symbol: str):
    try:
        symbol = symbol.upper()
        conn = psycopg2.connect(database=dbname, password=key, user=user, host=host, port=port)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM public.stock WHERE symbol = %s", (symbol,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result
    except Exception as e:
        st.error(f"Error searching stock: {e}")
        return None
    
def check_with_database(symbol: str):
    try:
        symbol = symbol.lower()
        conn = psycopg2.connect(database=dbname, password=key, user=user, host=host, port=port)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT symbol FROM public.stock WHERE LOWER(name) LIKE %s",
            (f"%{symbol}%",)
        )
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        if result:
            return result[0][0]
        else:
            return None
    except Exception as e:
        st.error(f"Error checking with database: {e}")
        return None
    
def add_stock_to_database(symbol: str, full_name: str):
    try:
        symbol = symbol.upper()
        # Escape single quotes in the full_name by replacing ' with ''
        full_name = full_name.replace("'", "''")
        
        client = psycopg2.connect(
            host=st.secrets['postgres']['host'],
            database=st.secrets['postgres']['db_name_postgres'],
            user=st.secrets['postgres']['user'],
            password=st.secrets['postgres']['password']
        )
        cursor = client.cursor()
        
        # Check if the stock already exists in the database
        cursor.execute("SELECT * FROM stock WHERE symbol = %s", (symbol,))
        existing_stock = cursor.fetchone()
        if existing_stock:
            st.warning(f"Stock {symbol} already exists in the database")
            return False
        
        # Use parameterized query to safely handle special characters
        cursor.execute(
            "INSERT INTO stock (symbol, name, instrument_type) VALUES (%s, %s, %s)",
            (symbol, full_name, 'equity')
        )
        
        client.commit()
        cursor.close()
        client.close()
        st.success(f"Stock {symbol} added to database")
        return True
    except Exception as e:
        st.error(f"Error adding stock {symbol} to database: {e}")
        return False

def add_crypto_to_database(symbol: str, full_name: str):
    try:
        symbol = symbol.upper()
        # Escape single quotes in the full_name by replacing ' with ''
        full_name = full_name.replace("'", "''")
        
        client = psycopg2.connect(
            host=st.secrets['postgres']['host'],
            database=st.secrets['postgres']['db_name_postgres'],
            user=st.secrets['postgres']['user'],
            password=st.secrets['postgres']['password']
        )
        cursor = client.cursor()
        
        # Check if the stock already exists in the database
        cursor.execute("SELECT * FROM stock WHERE symbol = %s", (symbol,))
        existing_stock = cursor.fetchone()
        if existing_stock:
            st.warning(f"Crypto {symbol} already exists in the database")
            return False
        
        # Use parameterized query to safely handle special characters
        cursor.execute(
            "INSERT INTO stock (symbol, name, instrument_type) VALUES (%s, %s, %s)",
            (symbol, full_name, 'crypto')
        )
        
        client.commit()
        cursor.close()
        client.close()
        st.success(f"Crypto {symbol} added to database")
        return True
    except Exception as e:
        return False

def check_symbol_yahoo(symbol):
    try:
        ticker = yf.Ticker(symbol)
        # Try to get current price - if successful, stock exists
        current_price = ticker.info.get('currentPrice')
        return current_price is not None
    except Exception as e:
        return False

def check_crypto_exists(symbol: str) -> tuple:
    """Check if a crypto symbol exists and return market type and ticker name."""
    
    try:
        symbol = symbol.upper()
        cryto_symbol = f"X:{symbol}USD"
        ticker_info = polygon_client.get_ticker_details(cryto_symbol)
        return symbol, ticker_info.base_currency_name
    
    except Exception as e:
        # Handle the case when symbol is not found
        return None, None
    
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
