import streamlit as st
from pymongo import MongoClient
from dependencies import init_postgres,init_mongodb_portfolio, verify_user, sign_up_process, forgot_password
from add_portfolio import add_portfolio
from Dashboard import user_dashboard
from long_term import long_term_dashboard
from short_term import short_term_dashboard
from settings_page import settings_page
import redis
import time
from openai import OpenAI
import json
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
            st.button("🚪 Log Out", on_click=logout)
            st.button("⚙️ Settings", on_click=lambda: st.session_state.update(current_page="Settings"))
            st.button("📊 Edit Portfolio", on_click=lambda: st.session_state.update(current_page="Portfolio"))
        with col2:
            st.button("🏠 User Dashboard", on_click=lambda: st.session_state.update(current_page="Main Page"))
            st.button("📈 Stock InDepth", on_click=lambda: st.session_state.update(current_page="Long Term"))
            st.button("⚡ Fast Money", on_click=lambda: st.session_state.update(current_page="Short Term"))
        
        chat_placeholder = st.sidebar.empty()
        with chat_placeholder:
            # Display chat header in the sidebar
            st.sidebar.markdown("""
                <div class="chat-header" style="
                    text-align: center;
                    padding: 15px;
                    background-color: #f0f8ff;
                    border-radius: 10px;
                    margin: 10px 0;
                    font-size: 18px;
                    font-weight: bold;
                    color: #2c3e50;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    ✨ Condvest Advisor Chat 💬
                </div>
            """, unsafe_allow_html=True)

            # Create a placeholder for the chat container in the sidebar
            chat_placeholder = st.sidebar.empty()

            with chat_placeholder:
                st.sidebar.markdown("""
                    <div class="chat-container">
                        <div class="chat-messages">
                """, unsafe_allow_html=True)
                
                st.sidebar.markdown("</div></div>", unsafe_allow_html=True)
                
            # Accept user input outside the chat_placeholder
            if prompt := st.sidebar.text_input("Ask me what is this about and how does it work?"):
                
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message in chat message container
                with st.sidebar.chat_message("user"):
                    st.markdown(f"""
                        <div style="width: 100%; word-wrap: break-word;">
                            {prompt}
                        </div>
                    """, unsafe_allow_html=True)

                # Simulate assistant response (replace with actual API call logic)
                with st.sidebar.chat_message("assistant"):
                    condvest_advisor = client.chat.completions.create(
                                        model="gpt-4o-mini",
                                        messages=st.session_state.messages,
                                        response_format={
                                            "type": "text"
                                        },
                                        temperature=1,
                                        max_tokens=2048,
                                        top_p=1,
                                        frequency_penalty=0,
                                        presence_penalty=0)
                    condvest_advisor_response = condvest_advisor.choices[0].message.content
                    # Display assistant's response in chat message container
                    st.markdown(f"""
                        <div style="width: 100%; word-wrap: break-word;">
                            {condvest_advisor_response}
                        </div>
                    """, unsafe_allow_html=True)
                
    elif st.session_state['current_page'] == "Sign Up":
        st.sidebar.button("Log in", on_click=lambda: st.session_state.update(current_page="Login"))
    elif st.session_state['current_page'] == "Forgot Password":
        st.sidebar.button("Back to Login", on_click=lambda: st.session_state.update(current_page="Login"))
    # Display page content based on the current page
    placeholder = st.empty()
    with placeholder.container():
        if st.session_state['current_page'] == "Login":
            col1, col2 = st.columns([2, 3])
            with col1:
                login()
                st.button("Sign Up", on_click=lambda: st.session_state.update(current_page="Sign Up"))
                st.button("Forgot Password?", on_click=lambda: st.session_state.update(current_page="Forgot Password"))
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
                        animation: scroll 480s linear infinite; 
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
                    f"<span style='color: {'#4CAF50' if item['final_profit_loss_pct'] > 0 else '#FF5733'}'>"
                    f"🚀 {item['symbol']}: Entry {item['entry_date'].strftime('%Y-%m-%d')} | Exit {item['exit_date'].strftime('%Y-%m-%d')} | "
                    f"Profit/Loss: {item['final_profit_loss_pct'] * 100:.2f}%"
                    f"</span>"
                    for item in sand_box_results if 'entry_date' in item and 'exit_date' in item
                )
                # Display scrolling marquee in Streamlit
                st.markdown(
                    f"""
                    <div class="marquee-container">
                        <span class="marquee">
                            CondVest Backtest Results from 2021: {scrolling_text}
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
    
