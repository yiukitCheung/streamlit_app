import streamlit as st
import psycopg2
import os, re, hashlib
from twilio.rest import Client
from dotenv import load_dotenv

# load_dotenv()

# Database connection details
dbname = st.secrets["db_name_postgres"]
user = st.secrets["user"]
host = st.secrets['host']
port = st.secrets['port']
key = st.secrets['password']

# account_sid = os.getenv('ACC_SID')
# auth_token = os.getenv('AUTH_TOKEN')

account_sid = st.secrets['twilio']['ACC_SID']
auth_token = st.secrets['twilio']['AUTH_TOKEN']

client = Client(account_sid, auth_token)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

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

    except psycopg2.Error as e:
        st.error(f"Database error: {e}")
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

def get_usernames():
    try:
        conn = psycopg2.connect(database=dbname, user=user, host=host, port=port)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM public.users")
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        return result
    except Exception as e:
        st.error(f"Error fetching usernames: {e}")
    return []

def get_phone_number():
    try:
        conn = psycopg2.connect(database=dbname, user=user, host=host, port=port)
        cursor = conn.cursor()
        cursor.execute("SELECT mobile FROM public.users")
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        return result
    except Exception as e:
        st.error(f"Error fetching phone numbers: {e}")
    return []

def validate_username(username):
    return bool(re.match(r'^[a-zA-Z0-9_]+$', username))

def validate_phone_number(phone_number):
    return bool(re.match(r'^[0-9]{10}$', phone_number))

def insert_user_data(username, password, phone_number):
    try:
        conn = psycopg2.connect(database=dbname, user=user, host=host, port=port, password=key)
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
        conn = psycopg2.connect(database=dbname, user=user, host=host, port=port, password=key)
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
