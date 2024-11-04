import streamlit as st
from pymongo import MongoClient
from dependencies import init_postgres, verify_user, sign_up_process
from Dashboard import user_dashboard
from long_term import long_term_dashboard
from short_term import display_live_data
import random
# Ensure the correct path to the 'data' directory
from trading_strategy import DailyTradingStrategy

# Database connection details
dbname = st.secrets["db_name_postgres"]
user = st.secrets["user"]
host = st.secrets['host']
port = st.secrets['port']

# MongoDB Config
URL = st.secrets['mongo'].host
DB_NAME = st.secrets.db_name
WAREHOUSE_INTERVAL = st.secrets.warehouse_interval
PROCESSED_COLLECTION = st.secrets.processed_collection_name
ALERT_COLLECTION = st.secrets.alert_collection_name
SANDBOX_COLLECTION = st.secrets.alert_sandbox_name

# Fetch Sandbox testing results
def fetch_sandbox_records():
    client = MongoClient(URL)
    db = client[DB_NAME]
    records = list(db[SANDBOX_COLLECTION].find({}, {'_id':0}))
    return records

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "Login"
if 'show_login' not in st.session_state:
    st.session_state['show_login'] = False

def login():
    st.header("Member Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login", key="login"):
        if not username or not password:
            st.error("Please enter both username and password.")
        else:
            is_valid, role = verify_user(username, password)
            if is_valid:
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.session_state['role'] = role
                st.session_state['current_page'] = "Dashboard" if role == 'admin' else "Main Page"
                st.success(f"Welcome {username}!")
                st.rerun()
            else:
                st.error("Invalid username or password")

def logout():
    for key in ['logged_in', 'username', 'role', 'current_page']:
        st.session_state.pop(key, None)
    st.success("Logged out successfully!")
    st.session_state['current_page'] = "Login"

def main():
    st.set_page_config(layout="wide")
    init_postgres()
    sand_box_results = fetch_sandbox_records()

    # Sidebar navigation buttons
    if st.session_state.get('logged_in'):
        st.sidebar.button("Log Out", on_click=logout)
        st.sidebar.button("User Dashboard", on_click=lambda: st.session_state.update(current_page="Main Page"))
        st.sidebar.button("Long Term Analysis", on_click=lambda: st.session_state.update(current_page="Long Term"))
        st.sidebar.button("Short Term Analysis", on_click=lambda: st.session_state.update(current_page="Short Term"))


    # Display page content based on the current page
    placeholder = st.empty()
    with placeholder.container():
        if st.session_state['current_page'] == "Login":
            col1, col2 = st.columns([2, 3])
            with col1:
                login()
                st.button("Sign Up", on_click=lambda: st.session_state.update(current_page="Sign Up"))
                # Display the marquee text
                # CSS for scrolling marquee effect
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
                        animation: scroll 70s linear infinite; 
                    }

                    /* Define the scrolling animation */
                    @keyframes scroll {
                        0% { transform: translateX(25%); } /* Start closer */
                        100% { transform: translateX(-100%); }
                    }
                    </style>
                """, unsafe_allow_html=True)

                # Create scrolling text with conditional colors for profit/loss
                scrolling_text = " | ".join(
                    f"<span style='color: {'#4CAF50' if item['final_profit_loss_pct'] > 0 else '#FF5733'}'>"
                    f"🚀 {item['symbol']}: Entry {item['entry_date'].strftime('%Y-%m-%d')} | Exit {item['exit_date'].strftime('%Y-%m-%d')} | "
                    f"Profit/Loss: {item['final_profit_loss_pct']:.2f}%"
                    f"</span>"
                    for item in sand_box_results if 'entry_date' in item and 'exit_date' in item
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
            with col2:
                st.markdown("""
                <div stype="background-color: #f9f9f9; padding: 20px; border-radius: 10px;">
                    <h2 style="color: #2E8B57;">About Us</h2>
                <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px;">
                    <h2 style="color: #2E8B57;">CondVest</h2>
                    <p style="font-size: 16px; line-height: 1.6;">
                        At CondVest, our mission is to simplify the complexities of the equity market for individual investors. 
                        We believe that everyone deserves access to clear, actionable insights without the noise that often surrounds market data. 
                        By integrating, analyzing, and distilling vast amounts of financial information, we provide investors with concise, easy-to-understand alerts and insights that truly matter.
                    </p>
                    <p style="font-size: 16px; line-height: 1.6;">
                        Our automated alert system is powered by advanced technical analysis, designed to spotlight potential opportunities, 
                        highlight risks, and track capital flows in the market. We reduce the need for extensive due diligence, 
                        helping investors make informed decisions without the time-consuming process of sorting through overwhelming data.
                    </p>
                    <h3 style="color: #2E8B57;">Our Principles</h3>
                    <ul style="font-size: 16px; line-height: 1.6;">
                        <li><b>Clarity Over Complexity:</b> We turn intricate market data into clear, valuable insights, empowering investors to act with confidence.</li>
                        <li><b>Responsible Investing:</b> At CondVest, we prioritize helping investors build structured, low-risk trading practices, cultivating disciplined investing habits.</li>
                        <li><b>Investor-Centric Approach:</b> Unlike other platforms, our goal is to support your success. We’re here to help you make profits, not just presenting information.</li>
                    </ul>
                    <h3 style="color: #2E8B57;">Our Vision</h3>
                    <p style="font-size: 16px; line-height: 1.6;">
                        We envision a world where individual investors have the tools they need to navigate the financial markets responsibly and profitably. 
                        By guiding investors to make smarter, well-informed decisions, CondVest aims to contribute to a more financially empowered society.
                    </p>
                </div>
                """, unsafe_allow_html=True)

        elif st.session_state['current_page'] == "Sign Up":
            sign_up_process()
        elif st.session_state['current_page'] == "Main Page":
            user_dashboard()
        elif st.session_state['current_page'] == "Long Term":
            long_term_dashboard()
        elif st.session_state['current_page'] == "Short Term":
            display_live_data()

if __name__ == "__main__":
    main()