import streamlit as st
import json
import redis
from openai import OpenAI
from pymongo import MongoClient
from dependencies import (
    init_postgres, init_mongodb_portfolio, 
    verify_user, sign_up_process, forgot_password
)
from add_portfolio import add_portfolio
from Dashboard import user_dashboard
from long_term import long_term_dashboard
from short_term import short_term_dashboard
from settings_page import settings_page
import gettext, os, time

# OpenAI Configuration
client = OpenAI(api_key=st.secrets['chatgpt']['api_key'])
assistant_id = st.secrets['chatgpt']['assistant_id']
messages = json.load(open('chat_message.json'))
# Database connection details
dbname = st.secrets["postgres"]["db_name_postgres"]
user = st.secrets["postgres"]["user"]
host = st.secrets['postgres']['host']
port = st.secrets['postgres']['port']

# MongoDB Config
URL = st.secrets['mongo']['host']
DB_NAME = st.secrets['mongo']['db_name']
WAREHOUSE_INTERVAL = st.secrets['mongo']['warehouse_interval']
PROCESSED_COLLECTION = st.secrets['mongo']['processed_collection_name']
ALERT_COLLECTION = st.secrets['mongo']['alert_collection_name']
SANDBOX_COLLECTION = st.secrets['mongo']['alert_sandbox_name']

# Redis Configuration
REDIS_HOST = st.secrets['redis']['host']
REDIS_PORT = st.secrets['redis']['port']
REDIS_PASSWORD = st.secrets['redis']['password']

def initialize_redis():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD)

# Fetch Sandbox testing results
def fetch_sandbox_records():
    client = MongoClient(URL)
    db = client[DB_NAME]
    records = list(db[SANDBOX_COLLECTION].find({}, {'_id':0}, sort=[('entry_date', 1)]))
    return records

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "Login"
if 'show_login' not in st.session_state:
    st.session_state['show_login'] = False
if 'instrument' not in st.session_state:
    st.session_state['instrument'] = "index"
if "profile_verified" not in st.session_state:
    st.session_state["profile_verified"] = False
if "alert_symbols" not in st.session_state:
    st.session_state["alert_symbols"] = []
if "messages" not in st.session_state:
    st.session_state["messages"] = messages
if "language" not in st.session_state:
    st.session_state["language"] = "en"

def logout():
    for key in ['logged_in', 'username', 'role', 'current_page']:
        st.session_state.pop(key, None)
    st.success("Logged out successfully!")
    st.session_state['current_page'] = "Login"

def main():
    # Set up Page Config only once
    st.set_page_config(layout="wide")
    
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

    # Set the default translation
    set_translation(lang)

    # Language Selector
    language = st.sidebar.radio("", ['English', '中文'])
    if language == '中文':
        lang = 'zh'
        st.session_state['language'] = 'zh'
    else:
        lang = 'en'
        st.session_state['language'] = 'en'

    # Update translation dynamically
    set_translation(lang)
    
    init_postgres()
    init_mongodb_portfolio()
    sand_box_results = fetch_sandbox_records()

    # Sidebar navigation buttons
    if st.session_state.get('logged_in'):
        st.markdown(
                """
                <style>
                    div[data-testid="stButton"] button {
                        width: 100%;
                        margin: 5px 0;
                        background-color: #f0f8ff;
                        color: #2c3e50;
                        border: 1px solid #e1e8ed;
                        border-radius: 10px;
                        padding: 10px;
                        font-weight: bold;
                        transition: all 0.3s;
                    }
                    div[data-testid="stButton"] button:hover {
                        background-color: #e1e8ed;
                        transform: translateY(-2px);
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    }
                </style>
                """,
                unsafe_allow_html=True
            )
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.button(_("🚪 Log Out"), on_click=logout)
            st.button(_("⚙️ Settings"), on_click=lambda: st.session_state.update(current_page="Settings"))
            st.button(_("📊 Edit Portfolio"), on_click=lambda: st.session_state.update(current_page="Portfolio"))
        with col2:
            st.button(_("🏠 User Dashboard"), on_click=lambda: st.session_state.update(current_page="Main Page"))
            st.button(_("📈 Stock InDepth"), on_click=lambda: st.session_state.update(current_page="Long Term"))
            st.button(_("⚡ Fast Money"), on_click=lambda: st.session_state.update(current_page="Short Term"))          
    elif st.session_state['current_page'] == "Sign Up":
        st.sidebar.button(_("Log in"), on_click=lambda: st.session_state.update(current_page="Login"))
    elif st.session_state['current_page'] == "Forgot Password":
        st.sidebar.button(_("Back to Login"), on_click=lambda: st.session_state.update(current_page="Login"))
    # Display page content based on the current page
    placeholder = st.empty()
    with placeholder.container():
        if st.session_state['current_page'] == "Login":
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
                    padding-left: 100%; /* Start outside view */
                    font-size: 18px;
                    font-weight: bold;
                    animation: scroll 60s linear infinite; 
                }

                /* Define the scrolling animation */
                @keyframes scroll {
                    100% { transform: translateX(15%); } /* Start closer */
                    100% { transform: translateX(-100%); }
                }
                </style>
            """, unsafe_allow_html=True)
            # Create scrolling text with conditional colors for profit/loss
            scrolling_text = " | ".join(
                f"<span style='color: {'#4CAF50' if float(item['profit/loss'].split('%')[0]) > 0 else '#FF5733'}'>"
                f"🚀 {item['symbol']}: Entry {item['Entry_date'].strftime('%Y-%m-%d')} | Exit {item['Exit_date'].strftime('%Y-%m-%d')} | "
                f"Profit/Loss: {float(item['profit/loss'].split('%')[0]):.2f}%"
                f"</span>"
                for item in sand_box_results if ('Entry_date' in item) and ('Exit_date' in item) and (float(item['profit/loss'].split('%')[0]) > 0)
            )
            # Display scrolling marquee in Streamlit
            st.markdown(
                f"""
                <div class="marquee-container">
                    <span class="marquee">
                        {_("CondVest Backtest Results from 2024:")} {scrolling_text}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
            # Split the page into two columns
            col1, col2 = st.columns([1, 3])
            with col1:
                
                # Add custom CSS for button styling
                st.markdown("""
                    <style>
                    div.stButton > button {
                        width: 100%;
                        margin: 5px 0;
                        background-color: white;
                        color: #4a8f4a;
                        border-radius: 8px;
                        border: none;
                        padding: 10px 20px;
                        font-weight: 900;
                        letter-spacing: 0.5px;
                        box-shadow: 0 4px 8px rgba(64, 224, 208, 0.2);
                        transform: translateY(0);
                        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                        text-transform: uppercase;
                        font-size: 14px;
                    }
                    div.stButton > button:hover {
                        box-shadow: 0 8px 16px rgba(64, 224, 208, 0.3);
                        transform: translateY(-2px);
                        background-color: rgba(64, 224, 208, 0.05);
                        font-weight: 900;
                    }
                    div.stButton > button:active {
                        transform: translateY(0);
                        box-shadow: 0 2px 4px rgba(64, 224, 208, 0.2);
                    }
                    </style>
                """, unsafe_allow_html=True)

                # Login section with styled header
                st.markdown(
                    f"<h2 style='text-align: center; color: #2c3e50;'>{_('Member Login')}</h2>",
                    unsafe_allow_html=True
                )                   
                # Input fields with consistent styling
                username = st.text_input(_("Username "), key="username_input", placeholder="admin") # Added space after Username to align with Password
                password = st.text_input(_("Password "), type="password", key="password_input", placeholder="1234") # Added space after Password

                # Login button with validation
                if st.button(_("Login"), key="login"):
                    if not username or not password:
                        st.error(_("Please enter both username and password."))
                    else:
                        is_valid, role = verify_user(username, password)
                        if is_valid:
                            st.session_state['logged_in'] = True
                            st.session_state['username'] = username
                            st.session_state['role'] = role
                            st.session_state['current_page'] = "Main Page"
                            st.success(_(f"Welcome {username}!"))
                            st.rerun()
                        else:
                            st.error(_("Invalid username or password"))
                # Additional buttons with consistent styling
                st.button(_("Sign Up"), on_click=lambda: st.session_state.update(current_page="Sign Up"))
                st.button(_("Forgot Password?"), on_click=lambda: st.session_state.update(current_page="Forgot Password"))
                if st.button(_("Please Check Us Out")):
                    st.session_state.update(current_page="Main Page")
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = "guest"
                    st.session_state['role'] = "guest"
                    st.rerun()
            with col2:
                st.markdown(_("""
                <div stype="background-color: #f9f9f9; padding: 20px; border-radius: 10px;">
                    <h2 style="color: #2E8B57;">{}</h2>
                <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px;">
                    <h2 style="color: #2E8B57;">{}</h2>
                    <p style="font-size: 16px; line-height: 1.6;">
                        {}
                    </p>
                    <p style="font-size: 16px; line-height: 1.6;">
                        {}
                    </p>
                    <h3 style="color: #2E8B57;">{}</h3>
                    <ul style="font-size: 16px; line-height: 1.6;">
                        <li><b>{}:</b> {}</li>
                        <li><b>{}:</b> {}</li>
                        <li><b>{}:</b> {}</li>
                    </ul>
                    <h3 style="color: #2E8B57;">{}</h3>
                    <p style="font-size: 16px; line-height: 1.6;">
                        {} 
                        {}
                    </p>
                </div>
                """).format(_("About Us"),
                            _("CondVest"),
                            _("At CondVest, our mission is to simplify the complexities of the equity market for individual investors. We believe that everyone deserves access to clear, actionable insights without the noise that often surrounds market data. By integrating, analyzing, and distilling vast amounts of financial information, we provide investors with concise, easy-to-understand alerts and insights that truly matter. By guiding investors to make smarter, well-informed decisions, CondVest aims to contribute to a more financially empowered society."),
                            _("Our automated alert system is powered by advanced technical analysis, designed to spotlight potential opportunities, highlight risks, and track capital flows in the market. We reduce the need for extensive due diligence, helping investors make informed decisions without the time-consuming process of sorting through overwhelming data."),
                            _("Our Mission"),
                            _("Clarity Over Complexity"),
                            _("We turn intricate market data into clear, valuable insights, empowering investors to act with confidence."),
                            _("Responsible Investing"),
                            _("At CondVest, we prioritize helping investors build structured, low-risk trading practices, cultivating disciplined investing habits."),
                            _("Investor-Centric Approach"),
                            _("Unlike other platforms, our goal is to support your success. We’re here to help you manage your portfolio, not just presenting information."),
                            _("Our Principles"),
                            _("We envision a world where individual investors have the tools they need to navigate the financial markets responsibly and profitably."),
                            _("By guiding investors to make smarter, well-informed decisions, CondVest aims to contribute to a more financially empowered society."),
                            _("Our Vision")),
                unsafe_allow_html=True)
                
        elif st.session_state['current_page'] == "Forgot Password":
            forgot_password()
        elif st.session_state['current_page'] == "Sign Up":
            sign_up_process()
        elif st.session_state['current_page'] == "Main Page":
            user_dashboard()
        elif st.session_state['current_page'] == "Long Term":
            long_term_dashboard()
        elif st.session_state['current_page'] == "Short Term":
            short_term_dashboard()
        elif st.session_state['current_page'] == "Portfolio":
            add_portfolio()
        elif st.session_state['current_page'] == "Settings":
            settings_page()

if __name__ == "__main__":
    main()
    
