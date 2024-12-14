import streamlit as st
import pandas as pd
import yfinance as yf
from pymongo import MongoClient
from dependencies import initialize_mongo_client, fetch_symbol_portfolio, add_stock_to_database, check_symbol_yahoo
import psycopg2
import gettext
import os

# MongoDB Configuration
DB_NAME = st.secrets['mongo']['db_name']
WAREHOUSE_INTERVAL = st.secrets['mongo']['warehouse_interval']
WAREHOUSE_INTERVAL_COLLECTION = '1d_data'
PROCESSED_COLLECTION = st.secrets['mongo']['processed_collection_name']
ALERT_COLLECTION = st.secrets['mongo']['alert_collection_name']
CANDI_COLLECTION = st.secrets['mongo']['candidate_collection_name']
PORTFOLIO_COLLECTION = st.secrets['mongo']['portfolio_collection_name']


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
        
def add_portfolio():
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
    username = st.session_state['username']
    
    # Display header based on portfolio status
    if existing_portfolio(username):
        header_text = _("Add Stock to Portfolio")
    else:
        header_text = _("Portfolio is Empty")
        
    st.markdown(f"""
        <div style="text-align: center; font-size: 24px; font-weight: bold; color: #2c3e50;">
            {header_text}
        </div>
    """, unsafe_allow_html=True)

    # Search and add stock section
    search_stock = st.text_input(_("🔍 Search Stock:"), placeholder=_("Enter stock symbol..."))
    
    if search_stock:
        search_result = fetch_symbol_portfolio(search_stock)
        
        if search_result is not None:
            # Stock found - show add form
            avg_price = st.text_input(_("💰 Enter Average Price:"), placeholder=_("e.g. 150.50"))
            shares = st.text_input(_("📈 Enter Number of Shares:"), placeholder=_("e.g. 100"))
            
            if st.button(_("➕ Add to Portfolio"), use_container_width=True):
                add_to_portfolio(search_stock, shares, avg_price, username)
                st.success(_("✅ Successfully Added to Portfolio!"))
                
        else:
            # Stock not found - show contribute option
            st.warning(_("😔 Sorry, Stock not found in database"))
            st.info(_("🤔 Would you like to contribute this stock to our database?"))
            
            if st.button(_("✨ Contribute New Stock"), use_container_width=True):
                if check_symbol_yahoo(search_stock.upper()):
                    search_stock = search_stock.upper()
                    full_name = yf.Ticker(search_stock).info.get('longName')
                    add_stock_to_database(search_stock, full_name)
                    st.success(_("🎉 Thanks for your contribution! Your stock will be added to the database tomorrow."))
                    st.balloons()
                else:
                    st.warning(_("😔 Sorry, this stock is not available on Market"))

    # Delete stock section
    st.markdown("""
        <div style="text-align: center; font-size: 24px; font-weight: bold; color: #2c3e50;">
            {}
        </div>
    """.format(_("Delete Stock from Portfolio")), unsafe_allow_html=True)
    
    stock_to_delete = st.text_input(_("Enter the stock to delete from portfolio:"))
    if st.button(_("🗑️ Delete from Portfolio"), use_container_width=True):
        delete_from_portfolio(stock_to_delete, username)