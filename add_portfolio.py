import streamlit as st
import pandas as pd
import yfinance as yf
from pymongo import MongoClient

# MongoDB Configuration
DB_NAME = st.secrets['mongo']['db_name']
WAREHOUSE_INTERVAL = st.secrets['mongo']['warehouse_interval']
WAREHOUSE_INTERVAL_COLLECTION = '1d_data'
PROCESSED_COLLECTION = st.secrets['mongo']['processed_collection_name']
ALERT_COLLECTION = st.secrets['mongo']['alert_collection_name']
CANDI_COLLECTION = st.secrets['mongo']['candidate_collection_name']
PORTFOLIO_COLLECTION = st.secrets['mongo']['portfolio_collection_name']

@st.cache_resource
def initialize_mongo_client():
    client = MongoClient(st.secrets["mongo"]["host"])
    return client

def fetch_symbol_portfolio(symbol: str):
    try:
        symbol = symbol.upper()
        collection_obj = initialize_mongo_client()[DB_NAME][PROCESSED_COLLECTION]
        cursor = collection_obj.find_one({'symbol': symbol}, projection={'symbol': True, '_id': False})
        
        if cursor:
            return pd.DataFrame(list(cursor))

        else:
            return None
    except Exception as e:
        return None

def add_to_portfolio(symbol: str, shares: int, avg_price: float, username: str):
    
    collection_obj = initialize_mongo_client()[DB_NAME][PORTFOLIO_COLLECTION]
    existing_customer = collection_obj.find_one({'username': username})
    symbol = symbol.upper()
    if existing_customer:
        # Update the existing user's portfolio with new stock data
        existing_customer['portfolio'][symbol] = {'shares': shares, 'avg_price': avg_price}
        collection_obj.update_one(
            {'username': username},
            {'$set': {'portfolio': existing_customer['portfolio']}}
        )
        st.success(f"Portfolio updated for {username} with {symbol}.")
    else:
        # Create a new document for the user with their first stock
        collection_obj.insert_one({
            'username': username,
            'portfolio': {
                symbol: {'shares': shares, 'avg_price': avg_price}
            }
        })
        st.success(f"New portfolio created for {username} with {symbol}.")

def delete_from_portfolio(symbol: str, username: str):
    collection_obj = initialize_mongo_client()[DB_NAME][PORTFOLIO_COLLECTION]
    existing_customer = collection_obj.find_one({'username': username})
    symbol = symbol.upper()
    if existing_customer:
        collection_obj.update_one(
            {'username': username}, 
            {'$unset': {f'portfolio.{symbol}': ""}}
        )
        st.success("✅ Successfully Deleted from Portfolio!")
    else:
        st.warning(f"No portfolio found for {username}.")
    
def existing_portfolio(username: str):
    collection_obj = initialize_mongo_client()[DB_NAME][PORTFOLIO_COLLECTION]
    try:
        portfolio = collection_obj.find_one({'username': username}, projection={'portfolio': True, '_id': False})
        if len(portfolio['portfolio']) >= 1:
            return True
        else:
            return False

    except Exception as e:
        return False

def check_symbol_yahoo(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info != None
    except Exception as e:
        return False


def add_portfolio():
    if existing_portfolio(st.session_state['username']):
        st.markdown("""
            <div style="text-align: center; font-size: 24px; font-weight: bold; color: #2c3e50;">
                Add Stock to Portfolio
            </div>
        """, unsafe_allow_html=True)
        
        search_stock = st.text_input("🔍 Search Stock:", placeholder="Enter stock symbol...")
        try:
            if search_stock:
                search_result = fetch_symbol_portfolio(search_stock)
                if search_result is not None:
                    avg_price = st.text_input("💰 Enter Average Price:", 
                                            placeholder="e.g. 150.50")
                    shares = st.text_input("📈 Enter Number of Shares:",
                                        placeholder="e.g. 100") 
                    if st.button("➕ Add to Portfolio", 
                            use_container_width=True):
                        add_to_portfolio(search_stock, shares, avg_price, st.session_state['username'])
                        st.success("✅ Successfully Added to Portfolio!")
                else:
                    st.warning("😔 Sorry, Stock not found in database")
                    st.info("🤔 Would you like to contribute this stock to our database?")
                    if st.button("✨ Contribute New Stock", use_container_width=True):
                        if check_symbol_yahoo(search_stock.upper()):
                            st.success("🎉 Thanks for your contribution! Our system is analyzing the stock and will add it to the database soon.")
                            st.balloons()
                        else:
                            st.warning("😔 Sorry, this stock is not available on Market")
            
        except TypeError as e:
            pass
        
        st.markdown("""
            <div style="text-align: center; font-size: 24px; font-weight: bold; color: #2c3e50;">
                Delete Stock from Portfolio
            </div>
        """, unsafe_allow_html=True)
        stock_to_delete = st.text_input("Enter the stock to delete from portfolio:")
        if st.button("🗑️ Delete from Portfolio", use_container_width=True):
            delete_from_portfolio(stock_to_delete, st.session_state['username'])

    else:
        username = st.session_state['username']
        st.markdown(f"""
            <div style="text-align: center; font-size: 28px; font-weight: bold; color: #666666; 
                margin: 30px 0; padding: 20px; border-radius: 10px; 
                background: rgba(240,240,240,0.3); box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                {username.capitalize()}'s Portfolio is Empty
            </div>
        """, unsafe_allow_html=True)
        search_stock = st.text_input("Search Stock:")
        avg_price = st.text_input("Enter the average price:")
        shares = st.text_input("Enter the total number of shares:")
        if search_stock and avg_price and shares:
            search_result = fetch_symbol_portfolio(search_stock)
            if not search_result.empty:
                if st.button("Add to Portfolio"):
                    add_to_portfolio(search_stock, shares, avg_price, st.session_state['username'])
                    st.write("Added to Portfolio")
        
        st.markdown("""
            <div style="text-align: center; font-size: 24px; font-weight: bold; color: #2c3e50;">
                Delete Stock from Portfolio
            </div>
        """, unsafe_allow_html=True)
        stock_to_delete = st.text_input("Enter the stock to delete from portfolio:")
        if st.button("🗑️ Delete from Portfolio", use_container_width=True):
            delete_from_portfolio(stock_to_delete, st.session_state['username'])
