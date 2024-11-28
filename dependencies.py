import streamlit as st
import psycopg2
import os, re, hashlib
import pandas as pd
from pymongo import MongoClient
from twilio.rest import Client
import yfinance as yf
# Database connection details
dbname = st.secrets["postgres"]["db_name_postgres"]
user = st.secrets["postgres"]["user"]
host = st.secrets['postgres']['host']
port = st.secrets['postgres']['port']
key = st.secrets['postgres']['password']

mongo_uri = st.secrets['mongo']['host']
db_name = st.secrets['mongo']['db_name']
portfolio_collection_name = st.secrets['mongo']['portfolio_collection_name']
sandbox_collection_name = st.secrets['mongo']['sandbox_collection_name']
processed_collection_name = st.secrets['mongo']['processed_collection_name']
account_sid = st.secrets['twilio']['ACC_SID']
auth_token = st.secrets['twilio']['AUTH_TOKEN']

client = Client(account_sid, auth_token)

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

    print(twil_phone_number)
    st.subheader('Sign Up')

    with st.form(key='signup_form', clear_on_submit=True):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
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
            if not validate_phone_number(phone_number):
                st.error("Phone number must be 10 digits.")
                return

            # Hash the password and insert user data
            hashed_pw = hash_password(password)
            if insert_user_data(username, hashed_pw, phone_number):
                st.success(f"Account created successfully for {username}!")
                send_welcome_msg(client, twil_phone_number, phone_number)
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

def validate_phone_number(phone_number):
    return bool(re.match(r'^[0-9]{10}$', phone_number))

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

def send_welcome_msg(twilio_client, number, target_number):
    body = 'Hi! Welcome to CondVest. The only financial market assistant that help you invest...'
    try:
        twilio_client.messages.create(
            body=body,
            from_=number,
            to=target_number
        )
        st.info(f"A welcoming message has been sent to the registered phone number")
    except Exception as e:
        st.error(f"Failed to send message: {e}")

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

def fetch_symbol_portfolio(symbol: str):
    try:
        symbol = symbol.upper()
        collection_obj = initialize_mongo_client()[db_name][processed_collection_name]
        cursor = collection_obj.find_one({'symbol': symbol}, projection={'symbol': True, '_id': False})
        
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
    
def add_stock_to_database(symbol: str, full_name: str):
    try:
        symbol = symbol.upper()
        client = psycopg2.connect(
            host=st.secrets['postgres']['host'],
            database=st.secrets['postgres']['db_name_postgres'],
            user=st.secrets['postgres']['user'],
            password=st.secrets['postgres']['password']
        )
        cursor = client.cursor()
        cursor.execute(f"INSERT INTO stock (symbol, name, instrument_type) VALUES ('{symbol}', '{full_name}', 'equity')")
        client.commit()
        cursor.close()
        client.close()
        st.success(f"Stock {symbol} added to database")
    except Exception as e:
        st.error(f"Error adding stock {symbol} to database: {e}")

def check_symbol_yahoo(symbol):
    try:
        ticker = yf.Ticker(symbol)
        # Try to get current price - if successful, stock exists
        current_price = ticker.info.get('currentPrice')
        return current_price is not None
    except Exception as e:
        return False
