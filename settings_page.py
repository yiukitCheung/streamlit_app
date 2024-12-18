import streamlit as st
import psycopg2
import numpy as np

POSTGRES_DB = st.secrets["postgres"]["db_name_postgres"]
POSTGRES_USER = st.secrets["postgres"]["user"]
POSTGRES_PASSWORD = st.secrets["postgres"]["password"]
POSTGRES_HOST = st.secrets["postgres"]["host"]
POSTGRES_PORT = st.secrets["postgres"]["port"]

def init_postgres():
    try:
        conn = psycopg2.connect(
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT
        )
        return conn
    except psycopg2.Error as e:
        st.error(f"Error connecting to PostgreSQL: {e}")
        return None

def check_alert_settings_col():
    try:
        conn = init_postgres()
        cursor = conn.cursor()
        table_name = "users"
        alert_column = "alert_settings"
        
        # Check if the column exists
        check_col_exist_query = f"SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = '{table_name}' AND column_name = '{alert_column}')"
        cursor.execute(check_col_exist_query)
        col_exists = cursor.fetchone()[0]
        
        # If the column does not exist, create it
        if not col_exists:
            create_col_query = f"ALTER TABLE {table_name} ADD COLUMN {alert_column} VARCHAR(255)"
            cursor.execute(create_col_query)
            conn.commit()
            
        cursor.close()
        conn.close()
    except Exception as e:
        st.error(f"Error checking alert settings column: {e}")

def add_alert_settings(alert_type: str, username: str):
    
    try:
        conn = init_postgres()
        table_name = "users"
        alert_column = "alert_settings"
        
        # Insert the alert settings into the column
        if conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE {table_name} SET {alert_column} = %s WHERE username = %s", (alert_type, username))
            conn.commit()
            cursor.close()
            conn.close()
            
    except Exception as e:
        st.error(f"Error adding alert settings: {e}")
def unsubscribe_all_alerts(username: str):
    try:
        conn = init_postgres()
        table_name = "users"
        alert_column = "alert_settings"
        cursor = conn.cursor()
        cursor.execute(f"UPDATE {table_name} SET {alert_column} = %s WHERE username = %s", (None, username))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        st.error(f"Error unsubscribing all alerts: {e}")

def settings_page():
    # Check if the alert settings column exists, if not create it
    check_alert_settings_col()
    st.title("Settings")
    st.subheader("Alert Settings")
    
    username = st.session_state['username']
    # Add alert settings to the database
    send_all_alerts = st.selectbox("Send All Alerts", ["Yes", "No"], key="send_all_alerts")
    if send_all_alerts == "Yes":
        add_alert_settings("all", username)   
        st.success("All alerts will be sent")
    if send_all_alerts == "No":
        unsubscribe_all_alerts(username)
        st.success("All alerts will not be sent")