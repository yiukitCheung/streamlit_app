import pandas as pd
import numpy as np
from pymongo import MongoClient, DESCENDING
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import logging
import streamlit as st
import os, sys
# Add project root to sys.path dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from config.mongdb_config import load_mongo_config

MONGO_URL = st.secrets['mongo']['host']
DB_NAME = st.secrets['mongo']['db_name']
LONG_TERM_ALERT_COLLECTION = st.secrets['mongo']['alert_collection_name']


class ExpectedReturnRiskAnalyzer:
    def __init__(self, mongo_url=MONGO_URL, db_name=DB_NAME, collection_name=LONG_TERM_ALERT_COLLECTION):
        self.mongo_url = mongo_url
        self.db_name = db_name
        self.collection_name = collection_name

    def fetch_long_term_alerts(self, symbol, interval):
        """
        Fetch and prepare data for a given stock symbol.
        Parameters:
        symbol (str): The stock symbol to fetch data for.
        
        Returns:
        list: A list containing the alert data.
        """
        try:
            lst = list(MongoClient(self.mongo_url)[self.db_name][self.collection_name]\
                .find({'symbol': symbol, 'interval': {'$gte': interval}},{'_id': 0})\
                .sort([('date', DESCENDING), ('interval', DESCENDING)])\
                    .limit(20))
        except Exception as e:
            st.error(f"Error fetching long term alerts for {symbol}: {e}")
            return []
        
        return lst

    def get_all_possible_values(self, symbol, interval):    
        # Fetch alert data for the symbol
        long_term_alerts_dict = self.fetch_long_term_alerts(symbol, interval)
        if len(long_term_alerts_dict) == 0:
            st.warning(f"No Data found for {symbol} yet, you probably just contributed this stock!")
            return False, False
        
        # Get daily alerts and current price
        daily_alert_dict = [entry for entry in long_term_alerts_dict if entry['interval'] == interval]
        current_price = daily_alert_dict[0]['close']
        
        # Initialize data structure to store values for each interval
        available_point = {}
        distinct_interval = set([entry['interval'] for entry in long_term_alerts_dict])
        
        # Process each alert entry
        for entry in long_term_alerts_dict:
            # Break if we have data for all intervals
            if len(available_point) == len(distinct_interval):
                break
                
            # Initialize storage for new interval
            if entry['interval'] not in available_point:
                available_point[entry['interval']] = {
                    # EMAs
                    '8ema': [], '13ema': [], '144ema': [], '169ema': [],
                    
                    # Kernel density estimation points
                    'top': [], 'bottom': [], 'second_top': [], 'second_bottom': [],
                    
                    # Fibonacci retracement levels
                    'fib_236': [], 'fib_382': [], 'fib_500': [], 
                    'fib_618': [], 'fib_786': [], 'fib_1236': [], 'fib_1382': []
                }
            else:
                continue

            # Store EMA values if they exist
            for ema in ['8ema', '13ema', '144ema', '169ema']:
                if entry[ema]:
                    available_point[entry['interval']][ema].append(entry[ema])
            
            # Store kernel density estimation points
            kde_points = entry['structural_area']['kernel_density_estimation']
            for point in ['top', 'bottom', 'second_top', 'second_bottom']:
                available_point[entry['interval']][point].append(kde_points[point])
            
            # Store fibonacci retracement levels
            fib_levels = entry['structural_area']['fibonacci_retracement']
            for level in ['fib_236', 'fib_382', 'fib_500', 'fib_618', 
                        'fib_786', 'fib_1236', 'fib_1382']:
                available_point[entry['interval']][level].append(fib_levels[level])

        return available_point, current_price

    def get_all_possible_trends(self, symbol, interval):    
        # Fetch alert data for the symbol
        long_term_alerts_dict = self.fetch_long_term_alerts(symbol, interval)
        if len(long_term_alerts_dict) == 0:
            st.warning(f"No Data found for {symbol} yet, you probably just contributed this stock!")
            return False, False
        
        # Initialize data structure to store values for each interval
        available_trends = {}
        distinct_interval = set([entry['interval'] for entry in long_term_alerts_dict])
        
        # Process each alert entry
        for entry in long_term_alerts_dict:
            # Break if we have data for all intervals
            if len(available_trends) == len(distinct_interval):
                break
                
            # Initialize storage for new interval
            if entry['interval'] not in available_trends:
                if 'alerts' in entry and 'velocity_alert' in entry['alerts']:
                    available_trends[entry['interval']] = entry['alerts']['velocity_alert']
                else:
                    available_trends[entry['interval']] = None
            else:
                continue
                    
        return available_trends
    
    def find_sup_res(self, symbol, interval):
        available_point, close_price = self.get_all_possible_values(symbol, interval)
        available_trends = self.get_all_possible_trends(symbol, interval)
        if not available_point or not close_price:
            return False, False
        
        expected_support = {}
        expected_resistance = {}

        # Find support and resistance levels
        intervals = sorted(available_point.keys())
        for interval in intervals:
            # Check EMAs
            ema_support, ema_resistance = self._check_ema_levels(available_point[interval], close_price)
            if ema_support != float('-inf') and 'emas' not in expected_support:
                expected_support['emas'] = ema_support
            if ema_resistance != float('inf') and 'emas' not in expected_resistance:
                expected_resistance['emas'] = ema_resistance
            
            # Check structural areas
            area_support, area_resistance = self._check_structural_areas(available_point[interval], close_price)
            if area_support != float('-inf') and 'dense_area' not in expected_support:
                expected_support['dense_area'] = area_support
            if area_resistance != float('inf') and 'dense_area' not in expected_resistance:
                expected_resistance['dense_area'] = area_resistance

            # Check fibonacci levels
            fib_support, fib_resistance = self._check_fibonacci_levels(available_point[interval], close_price)
            if fib_support != float('-inf') and 'fibonacci' not in expected_support:
                expected_support['fibonacci'] = fib_support
            if fib_resistance != float('inf') and 'fibonacci' not in expected_resistance:
                expected_resistance['fibonacci'] = fib_resistance
        st.write(expected_support, expected_resistance)
        return expected_support, expected_resistance

    def _check_ema_levels(self, interval_data, close_price):
        # Check if support and resistance is in the short term ema
        min_ema = min(interval_data['13ema'][0], interval_data['8ema'][0])
        support = min_ema if close_price > min_ema else float('-inf')
        resistance = min_ema if close_price < min_ema else float('inf')
        
        # If support and resistance is not in the short term ema, check the long term ema
        if support == float('-inf'):
            # Check if the support is the 144ema or 169ema
            min_ema = min(interval_data['144ema'][0], interval_data['169ema'][0])
            support = min_ema if close_price > min_ema else float('-inf')
                
        if resistance == float('inf'):
            # Check if the resistance is the 144ema or 169ema
            max_ema = max(interval_data['144ema'][0], interval_data['169ema'][0])
            resistance = max_ema if close_price < max_ema else float('inf')
            
        return support, resistance

    def _check_structural_areas(self, interval_data, close_price):
        structural_areas = (
            interval_data['second_top'][0],
            interval_data['second_bottom'][0],
            interval_data['bottom'][0],
            interval_data['top'][0]
        )
        
        support = float('-inf')
        resistance = float('inf')
        
        for area in sorted(structural_areas):
            if close_price > area and (support is None or area > support):
                support = area
            elif close_price < area and (resistance is None or area < resistance):
                resistance = area
                
        return support, resistance

    def _check_fibonacci_levels(self, interval_data, close_price):
        fibonacci_levels = (
            interval_data['fib_236'][0],
            interval_data['fib_382'][0],
            interval_data['fib_500'][0],
            interval_data['fib_618'][0],
            interval_data['fib_786'][0],
            interval_data['fib_1236'][0],
            interval_data['fib_1382'][0]
        )
        
        support = float('-inf')
        resistance = float('inf')
        
        for level in sorted(fibonacci_levels):
            if close_price > level and (support is None or level > support):
                support = level
            elif close_price < level and (resistance is None or level < resistance):
                resistance = level
                
        return support, resistance
