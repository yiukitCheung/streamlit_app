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
from utils import StrategyEDA
import pandas as pd
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

@st.cache_resource
def initialize_mongo_client():
    try:
        client = MongoClient(st.secrets["mongo"]["host"])
    except Exception as e:
        st.error(f"Error initializing MongoDB client: {e}")
        return None
    return client

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

# Fetch Sandbox testing results
def fetch_sandbox_records():
    try:    
        client = MongoClient(URL)
        db = client[DB_NAME]
        records = list(db[SANDBOX_COLLECTION].find({}, {'_id':0}, sort=[('entry_date', 1)]))
        return records
    except Exception as e:
        st.error(f"Error fetching sandbox records: {e}")
        return []
    
def fetch_alert_data_last_3_candles():
    current_date = get_most_current_trading_date()
    try:
        collection_obj = initialize_mongo_client()[DB_NAME][PROCESSED_COLLECTION]
        query = {"date": {"$gte": pd.to_datetime(current_date) - pd.Timedelta(days=3)}}
        cursor = collection_obj.find(query)
        return list(cursor)
    except Exception as e:
        st.error(f"System Error, Please wait...: {str(e)}")
        return []

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
    st.sidebar.image("logo.png", use_column_width="auto")
    st.sidebar.markdown("<div style='text-align: center;'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='color: black; font-size: 18px; margin-bottom: 0; text-align: center;'>Language</p>", unsafe_allow_html=True)
    language = st.sidebar.selectbox("", ['English', 'French', 'Spanish', '中文'], key="language_selector", label_visibility="collapsed")
    if language == '中文':
        lang = 'zh'
        st.session_state['language'] = 'zh'
    elif language == 'French':
        lang = 'fr'
        st.session_state['language'] = 'fr'
    elif language == 'Spanish':
        lang = 'es'
        st.session_state['language'] = 'es'
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
        st.sidebar.button(_("Log in"), on_click=lambda: st.session_state.update(current_page="Login"), key="login_button")
    elif st.session_state['current_page'] == "Forgot Password":
        st.sidebar.button(_("Back to Login"), on_click=lambda: st.session_state.update(current_page="Login"), key="login_button")
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
                f"Profit/Loss: {float(item['profit/loss'].split('%')[0])/100:.2f}%"
                f"</span>"
                for item in sand_box_results if ('Entry_date' in item) and ('Exit_date' in item) and (float(item['profit/loss%'].split('%')[0]) > 10) and (item['instrument'] == 'equity')
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
                    f"<h2 style='text-align: center; color: #2E8B57;'>{_('Member Login')}</h2>",
                    unsafe_allow_html=True
                )                   
                # Input fields with consistent styling
                username = st.text_input(_("Username "), key="username_input") # Added space after Username to align with Password
                password = st.text_input(_("Password "), type="password", key="password_input") # Added space after Password
                
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
                st.markdown("""
                    <style>
                    .demo-button {
                        font-size: 20px !important;
                        padding: 15px 25px !important;
                        background: linear-gradient(45deg, rgba(46, 139, 87, 0.05), rgba(46, 139, 87, 0.15)) !important;
                        border: 2px solid #2E8B57 !important;
                        text-align: center !important;
                        color: #2E8B57 !important;
                        border-radius: 8px !important;
                        transition: all 0.3s ease !important;
                    }
                    .demo-button:hover {
                        background: linear-gradient(45deg, rgba(46, 139, 87, 0.15), rgba(46, 139, 87, 0.25)) !important;
                        box-shadow: 0 4px 12px rgba(46, 139, 87, 0.2) !important;
                        transform: translateY(-2px) !important;
                    }
                    </style>
                """, unsafe_allow_html=True)
                if st.button(_("Please Check Us Out"), key="demo_button", help=_("Experience CondVest without an account!!"), type="primary"):
                    st.session_state.update(current_page="Main Page")
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = "guest"
                    st.session_state['role'] = "guest"
                    st.rerun()
                    
            with col2:
                # Get the start and end date of the backtest results
                import pandas as pd
                st.plotly_chart(StrategyEDA(start_date = pd.to_datetime("2020-01-01"), end_date = pd.to_datetime("today"), instrument="equity").plot_trading_analysis(sand_box_results), use_container_width=True)       
                
                st.markdown("<br><br>", unsafe_allow_html=True)
            
            intro_col, mission_col, principle_col = st.columns([1, 1, 1])
            with intro_col:
                st.markdown(_("""
                    <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px;">
                        <h2 style="color: #2E8B57;">{}</h2>
                        <p style="font-size: 16px; line-height: 1.6;">
                            {}
                        </p>
                        <p style="font-size: 16px; line-height: 1.6;">
                            {}
                        </p>
                    </div>
                """).format(
                    _("About Us"),
                    _("At CondVest, our mission is to simplify the complexities of the equity market for individual investors. We believe that everyone deserves access to clear, actionable insights without the noise that often surrounds market data. By integrating, analyzing, and distilling vast amounts of financial information, we provide investors with concise, easy-to-understand alerts and insights that truly matter. By guiding investors to make smarter, well-informed decisions, CondVest aims to contribute to a more financially empowered society."),
                    _("Our automated alert system is powered by advanced technical analysis, designed to spotlight potential opportunities, highlight risks, and track capital flows in the market. We reduce the need for extensive due diligence, helping investors make informed decisions without the time-consuming process of sorting through overwhelming data.")
                ), unsafe_allow_html=True)

            with mission_col:
                st.markdown(_("""
                    <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px;">
                        <h2 style="color: #2E8B57;">{}</h2>
                        <ul style="font-size: 16px; line-height: 1.6;">
                            <li><b>{}:</b> {}</li>
                            <li><b>{}:</b> {}</li>
                            <li><b>{}:</b> {}</li>
                        </ul>
                    </div>
                """).format(
                    _("Our Mission"),
                    _("Clarity Over Complexity"), _("We turn intricate market data into clear, valuable insights, empowering investors to act with confidence."),
                    _("Responsible Investing"), _("At CondVest, we prioritize helping investors build structured, low-risk trading practices, cultivating disciplined investing habits."),
                    _("Investor-Centric Approach"), _("Unlike other platforms, our goal is to support your success. We’re here to help you manage your portfolio, not just presenting information.")
                ), unsafe_allow_html=True)

            with principle_col:
                st.markdown(_("""
                    <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px;">
                        <h2 style="color: #2E8B57;">{}</h2>
                        <p style="font-size: 16px; line-height: 1.6;">
                            {}
                        </p>
                        <p style="font-size: 16px; line-height: 1.6;">
                            {}
                        </p>
                    </div>
                """).format(
                    _("Our Vision"),
                    _("We envision a world where individual investors have the tools they need to navigate the financial markets responsibly and profitably."),
                    _("By guiding investors to make smarter, well-informed decisions, CondVest aims to contribute to a more financially empowered society.")
                ), unsafe_allow_html=True)
            
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
    
