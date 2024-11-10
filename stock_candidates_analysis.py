import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import os, sys
# Add project root to sys.path dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from config.mongdb_config import load_mongo_config

class Pick_Stock:

    def __init__(self, mongo_config, start_date=None, sandbox_mode=False):
        # Store MongoDB configuration
        self.mongo_config = mongo_config

        # Set Sandbox mode and start date accordingly
        self.sandbox_mode = sandbox_mode
        self.start_date = start_date if self.sandbox_mode else '2024-01-01'

        # Connect to MongoDB
        self.client = MongoClient(self.mongo_config['url'])
        self.db = self.client[self.mongo_config['db_name']]
        self.collection = self.db[self.mongo_config['alert_collection_name']['long_term']]
        self.candidate_collection_name = self.mongo_config['candidates_collection_name']['long_term']
        self.candidate_collection = self.db[self.candidate_collection_name]
        # Initialize a one hot encoder
        self.ohe = OneHotEncoder(sparse_output=False)
        # Dictionary to store stock candidates
        self.stock_candidates = {}

    def get_stock_dataframe(self):
        # Fetch the data from MongoDB and convert to DataFrame
        alert_dict = list(self.collection.find(
            {'date': {'$gte': pd.to_datetime(self.start_date)}},
            {'_id': 0}  # Exclude MongoDB's default '_id' field
        ))

        # Extract the alerts from the alert_dict and process fields
        for row in alert_dict:
            if 'alerts' in row and 'momentum_alert' in row['alerts']:
                row['momentum_alert'] = row['alerts']['momentum_alert']['alert_type']

            if 'alerts' in row and 'velocity_alert' in row['alerts']:
                row['velocity_alert'] = row['alerts']['velocity_alert']['alert_type']

            if 'alerts' in row and '169ema_touched' in row['alerts']:
                row['touch_type'] = row['alerts']['169ema_touched']['type']
                row['count'] = row['alerts']['169ema_touched']['count']

            elif 'alerts' in row and '13ema_touched' in row['alerts']:
                row['touch_type'] = row['alerts']['13ema_touched']['type']
                row['count'] = row['alerts']['13ema_touched']['count']

            else:
                row['touch_type'] = np.nan

        # Convert the alert_dict to a DataFrame and process it
        data = pd.DataFrame(alert_dict).drop(columns=['alerts'], errors='ignore')

        # Prepare the data for encoding
        alert_columns = ['touch_type', 'momentum_alert']

        encoded_arr = self.ohe.fit_transform(data[alert_columns])
        encoded_df = pd.DataFrame(encoded_arr, columns=self.ohe.get_feature_names_out())

        # Concat the encoded DataFrame with the original DataFrame
        data = pd.concat([data.drop(columns=alert_columns), encoded_df], axis=1)

        return data

    def create_time_series_collection(self, collection_name, keep_duration=None):
        if collection_name not in self.db.list_collection_names():
            self.db.create_collection(
                collection_name,
                timeseries={
                    "timeField": "date" if 'datastream' not in collection_name else "datetime",
                    "metaField": "symbol",
                    "granularity": "hours" if 'datastream' not in collection_name else "minutes"
                },
                expireAfterSeconds=keep_duration
            )
            logging.info(f"Time Series Collection {collection_name} created successfully")

    def insert_candidates(self, candidates):
        """
        Insert candidate stock data into a MongoDB time series collection, setting up auto-expiry for 7 days.
        Only insert data for dates that do not already exist.
        """
        # Create the time series collection if it doesn't exist
        keep_duration = 604800
        self.create_time_series_collection(self.candidate_collection_name, keep_duration=keep_duration)

        # Get the latest date in the collection
        latest_entry = self.candidate_collection.find_one(
            sort=[('date', -1)],
            projection={'date': 1}
        )
        # Check if there's an existing latest entry
        if latest_entry:
            latest_date = latest_entry['date']
        else:
            latest_date = pd.Timestamp.min
        # Prepare data for insertion, filtering out existing dates
        documents = []
        for date_str, values in candidates.items():
            logging.info(date_str)
            date_obj = pd.to_datetime(date_str)
            # Only insert data with dates later than the latest existing date
            if date_obj > latest_date:
                documents.append({
                    'date': date_obj,
                    'accelerating': values['accelerating'],
                    'main_accumulating': values['main_accumulating'],
                    'long_accelerating': values['long_accelerating'],
                    'long_main_accumulating': values['long_main_accumulating'],
                    'ext_long_accelerating': values['ext_long_accelerating'],
                    'ext_accumulating': values['ext_accumulating']
                })

        # Insert documents into MongoDB if there are new entries to add
        if documents:
            self.candidate_collection.insert_many(documents)
            logging.info("Candidates Stock inserted successfully!")
        else:
            logging.info("No new data to insert. All candidate data is up to date.")

    def evaluate_stocks(self, data):
        # Group by symbol and apply interval-based weighting for velocity alerts
        eval_sum = data.copy()

        # Apply Linear Weighting on Alert Data
        eval_sum['interval_weight'] = eval_sum['interval'].map(self.interval_weights)

        # Calculate weighted values for each alert type
        alerts_cols = ['touch_type_resistance', 'touch_type_support', 'momentum_alert_accelerated',
                    'momentum_alert_decelerated']
        for alert in alerts_cols:
            eval_sum[f'weighted_{alert}'] = eval_sum[alert] * eval_sum['interval_weight']

        eval_sum = eval_sum.drop(
            columns=alerts_cols + ['touch_type_nan', 'momentum_alert_nan', 'touch_type_neutral', 'interval_weight'])

        # Data Analysis
        results = eval_sum \
                    .loc[:, ['symbol',
                            'interval',
                            'weighted_momentum_alert_accelerated',
                            'weighted_momentum_alert_decelerated',
                            'weighted_touch_type_resistance',
                            'weighted_touch_type_support',
                            'count'
                            ]] \
            .groupby(['symbol', 'interval']) \
            .sum() \
            .sort_values(['interval', 'weighted_momentum_alert_accelerated', 'weighted_momentum_alert_decelerated',
                        'weighted_touch_type_support', 'count'],
                        ascending=[True, False, True, False, False]).reset_index()

        # Store the accelerating stock
        short_acc_equ = results[(results['weighted_momentum_alert_accelerated'] > 0) \
                                & (results['weighted_momentum_alert_decelerated'] < 1) \
                                & (results['interval'] <= 3)].loc[:, 'symbol']

        lng_acc_equ = results[(results['weighted_momentum_alert_accelerated'] > 0) \
                            & (results['weighted_momentum_alert_decelerated'] < 1) \
                            & (results['interval'] == 5)].loc[:, 'symbol']

        ext_lng_acc_equ = results[(results['weighted_momentum_alert_accelerated'] > 0) \
                                & (results['weighted_momentum_alert_decelerated'] < 1) \
                                & (results['interval'] == 13)].loc[:, 'symbol']

        # Store the main force accumulating stock
        short_main_acc_equ = results[(results['weighted_touch_type_support'] > 0) \
                                    & (results['weighted_touch_type_resistance']) < 1 \
                                    & (results['count'] > 2) \
                                    & (results['interval'] <= 3)].loc[:, 'symbol']

        lng_main_acc_equ = results[(results['weighted_touch_type_support'] > 0) \
                                & (results['weighted_touch_type_resistance']) < 1 \
                                & (results['count'] > 2) \
                                & (results['interval'] == 5)].loc[:, 'symbol']

        ext_lng_main_acc_equ = results[(results['weighted_touch_type_support'] > 0) \
                                    & (results['weighted_touch_type_resistance']) < 1 \
                                    & (results['count'] > 2) \
                                    & (results['interval'] == 13)].loc[:, 'symbol']

        # Create dictionary of results
        stock_dict = {
            'accelerating': short_acc_equ.tolist(),
            'main_accumulating': short_main_acc_equ.tolist(),
            'long_accelerating': lng_acc_equ.tolist(),
            'long_main_accumulating': lng_main_acc_equ.tolist(),
            'ext_long_accelerating': ext_lng_acc_equ.tolist(),
            'ext_accumulating': ext_lng_main_acc_equ.tolist(),
        }

        return stock_dict

    def run(self):
        # Fetch the data, sorted by symbol and date
        self.data = self.get_stock_dataframe()
        self.distinct_intervals = self.data['interval'].unique()
        # Define weights for intervals (e.g., larger intervals get more weight)
        self.interval_weights = {interval: weight for weight, interval in enumerate(self.distinct_intervals, start=1)}

        # Initialize a dictionary to hold the results with dates as keys
        candidates_dict = {}
        logging.info('generating stock candidates for each day...')
        for date in self.data['date'].unique():
            # Process and analyze the candidates stock of the day
            today_data = self.data[self.data['date'] == date]
            today_candidates = self.evaluate_stocks(today_data)

            # Assign today's candidates to the dictionary with date as the key
            candidates_dict[str(date)] = today_candidates  # Ensure date is string for JSON compatibility

        self.insert_candidates(candidates_dict)

class ComputeAcceleratingProfits:
    def __init__(self, mongo_config, start_date=None):
        self.mongo_config = mongo_config
        self.df = None
        self.alert_df = None
        self.stock_candidates = None

        # Set the start date to the first date of the current year if not provided
        if start_date is None:
            current_year = datetime.now().year
            self.start_date = f'{current_year}-01-01'
        else:
            self.start_date = start_date

    def load_data(self):
        client = MongoClient(self.mongo_config['url'])
        db = client[self.mongo_config['db_name']]
    
        # Load the processed data from the database
        self.df = pd.DataFrame(list(db[self.mongo_config['process_collection_name']].find(
            {'date': {'$gte': pd.to_datetime(self.start_date)}}
        )))
        
        # Load the stock candidates from the database
        self.stock_candidates = pd.DataFrame(list(db[self.mongo_config['candidates_collection_name']['long_term']].find(
            {'date': {'$gte': pd.to_datetime(self.start_date)}}
        )))

        # Load the alert data from the database
        alert_dict = list(db[self.mongo_config['alert_collection_name']['long_term']].find(
            {'date': {'$gte': pd.to_datetime(self.start_date)}}
        ))

        # Extract the alerts from the alert_dict
        for entry in alert_dict:
            if 'alerts' in entry and 'momentum_alert' in entry['alerts']:
                entry['momentum_alert'] = entry['alerts']['momentum_alert']['alert_type']
            if 'alerts' in entry and 'velocity_alert' in entry['alerts']:
                entry['velocity_alert'] = entry['alerts']['velocity_alert']['alert_type']
            if 'alerts' in entry and '169ema_touched' in entry['alerts']:
                entry['touch_type'] = entry['alerts']['169ema_touched']['type']
                entry['count'] = entry['alerts']['169ema_touched']['count']
            elif 'alerts' in entry and '13ema_touched' in entry['alerts']:
                entry['touch_type'] = entry['alerts']['13ema_touched']['type']
                entry['count'] = entry['alerts']['13ema_touched']['count']
            else:
                entry['touch_type'] = np.nan
                entry['count'] = 0.0

        # Convert the alert_dict to a DataFrame
        self.alert_df = pd.DataFrame(alert_dict).drop(columns=['alerts', '_id'])

        client.close()

    def find_alert(self, stock_data, desired_alerts):
        for alert in desired_alerts:
            cur_alert_data = stock_data.loc[alert]
            if len(cur_alert_data) != 0:
                return stock_data[alert][0]
            
    def compute_accelerating_profits(self):
        results = []

        for idx in range(len(self.df['date'].unique())):
            cur_stock_pick = self.stock_candidates.iloc[idx]
            stock = self.find_alert(cur_stock_pick, desired_alerts=['accelerating'])

            if not stock:
                continue

            # Get stock-specific data once
            stock_data = self.df[(self.df['symbol'] == stock) & (self.df['interval'] == 1)]

            if stock_data.empty:
                continue

            peak_profit_pct = 0
            dynamic_protection = False
            protected = False
            
            buy_date_idx = idx + 1
            if buy_date_idx >= len(stock_data):
                continue
                
            entry_date = stock_data.iloc[buy_date_idx]['date']
            entry_price = stock_data.iloc[buy_date_idx]['open']
            
            logging.info(f"buy {stock} at {entry_date}, price: {entry_price}")
            
            # Track the stock from the day after purchase
            available_hold_start = buy_date_idx + 1
            for future_date_idx in range(available_hold_start, len(stock_data)):
                future_date = stock_data.iloc[future_date_idx]['date']
                
                # Get velocity alert for the specific date and stock
                velocity_signals = self.alert_df[
                    (self.alert_df['symbol'] == stock) & 
                    (self.alert_df['date'] == future_date) &
                    (self.alert_df['interval'] == 1)  # Only consider interval 1 signals
                ]
                
                if velocity_signals.empty:
                    continue
                    
                signal = velocity_signals['velocity_alert'].iloc[0]
                current_price = stock_data.iloc[future_date_idx]['close']
                current_profit_pct = (current_price - entry_price) / entry_price
                
                # Dynamic protection logic
                if not dynamic_protection:
                    if current_profit_pct >= 0.3:  # 30% profit trigger
                        peak_profit_pct = current_profit_pct
                        logging.info(f"{stock}: Activating protection at {current_profit_pct:.2%} profit")
                        dynamic_protection = True
                        
                elif dynamic_protection:
                    if current_profit_pct > peak_profit_pct:
                        peak_profit_pct = current_profit_pct
                        logging.info(f"{stock}: New peak profit {peak_profit_pct:.2%}")
                        
                    # Check if profit dropped 50% from peak
                    if peak_profit_pct - current_profit_pct >= peak_profit_pct * 0.5:
                        logging.info(f'{stock}: Protection triggered sell at {current_profit_pct:.2%} (Peak was {peak_profit_pct:.2%})')
                        results.append({
                            'entry_date': entry_date,
                            'exit_date': future_date,
                            'symbol': stock,
                            'final_profit_loss_pct': current_profit_pct,
                            'exit_reason': 'protection'
                        })
                        break
                
                # Regular sell signals
                if not protected:
                    if signal == 'velocity_loss':
                        logging.info(f'{stock}: Velocity loss triggered sell at {current_profit_pct:.2%} in {future_date}')
                        results.append({
                            'entry_date': entry_date,
                            'exit_date': future_date,
                            'symbol': stock,
                            'final_profit_loss_pct': current_profit_pct,
                            'exit_reason': 'velocity_loss'
                        })
                        break
                    
                    # End of available data
                    if future_date_idx == len(stock_data) - 1:
                        logging.info(f'{stock}: End of data sell at {current_profit_pct:.2%}')
                        results.append({
                            'entry_date': entry_date,
                            'exit_date': future_date,
                            'symbol': stock,
                            'final_profit_loss_pct': current_profit_pct,
                            'exit_reason': 'end_of_data'
                        })
                        break

        return pd.DataFrame(results)

    def insert_results(self, results_df):
        client = MongoClient(self.mongo_config['url'])
        db = client[self.mongo_config['db_name']]
        collection_name = 'sandbox_results'

        if collection_name not in db.list_collection_names():
            db.create_collection(
                collection_name,
                timeseries={
                    "timeField": "entry_date",
                    "metaField": "symbol",
                    "granularity": "hours"
                }
            )
            logging.info(f"Time Series Collection {collection_name} created successfully")

        latest_record = db[collection_name].find_one(sort=[("entry_date", -1)])

        if latest_record:
            latest_date = latest_record['entry_date']
            results_df = results_df[results_df['entry_date'] > latest_date]
        else:
            logging.info("No existing records found. Inserting all results.")

        documents = results_df.to_dict(orient='records')

        if documents:
            try:
                db[collection_name].insert_many(documents, ordered=False)
                logging.info("Results inserted successfully!")
            except Exception as e:
                logging.error(f"Error inserting documents: {e}")
        else:
            logging.info("No new data to insert. All candidate data is up to date.")

        client.close()

    def run(self):
        self.load_data()
        results_df = self.compute_accelerating_profits()
        self.insert_results(results_df)

class DailyTradingStrategy:
    def __init__(self, mongo_config, start_date=None, sandbox_mode=False, initial_capital=10000, aggressive_split=1.0):
        self.protected = None
        self.peak_profit_pct = None
        self.peak_profit = None
        self.trades = []
        self.current_trade = {"conservative": {}, "aggressive": {}}
        self.start_date = start_date if sandbox_mode else '2024-01-01'

        self.aggressive_capital = initial_capital * aggressive_split
        self.conservative_capital = initial_capital * (1.0 - aggressive_split)
        self.capital_split = aggressive_split
        self.mongo_config = mongo_config
        self.dynamic_protection = False

        # Initialize MongoDB client and collections
        self.client = MongoClient(self.mongo_config['url'])
        self.db = self.client[self.mongo_config['db_name']]
        self.data_collection = self.db[self.mongo_config['process_collection_name']]
        self.alert_collection = self.db[self.mongo_config['alert_collection_name']['long_term']]
        self.stock_candidates = pd.DataFrame(list(self.db[self.mongo_config['candidates_collection_name']['long_term']] \
                                                .find({'date': {'$gte': pd.to_datetime(self.start_date)}},
                                                        {'_id': 0})))
        # Load data and alerts
        self.df = pd.DataFrame(list(self.data_collection. \
                                    find({'date': {'$gte': pd.to_datetime(self.start_date)}},
                                        {'_id': 0}))) \
            .sort_values(by=['symbol', 'date'])

        self.alert_df = self.get_alert_dataframe()

    def get_alert_dataframe(self):
        data_dict = list(self.alert_collection.find({'date': {'$gte': pd.to_datetime(self.start_date)}}, {'_id': 0}))
        for row in data_dict:
            if 'alerts' in row and 'momentum_alert' in row['alerts']:
                row['momentum_alert'] = row['alerts']['momentum_alert']['alert_type']
            if 'alerts' in row and 'velocity_alert' in row['alerts']:
                row['velocity_alert'] = row['alerts']['velocity_alert']['alert_type']
            if 'alerts' in row and '169ema_touched' in row['alerts']:
                row['touch_type'] = row['alerts']['169ema_touched']['type']
                row['count'] = row['alerts']['169ema_touched']['count']
            elif 'alerts' in row and '13ema_touched' in row['alerts']:
                row['touch_type'] = row['alerts']['13ema_touched']['type']
                row['count'] = row['alerts']['13ema_touched']['count']
            else:
                row['touch_type'] = np.nan
        return pd.DataFrame(data_dict).drop(columns=['alerts'])

    # Execute both types of trades
    def execute_critical_trades(self):

        for idx, date in enumerate(self.df['date'].unique()):
            # Handle aggressive trades
            if self.aggressive_capital >= 0:
                self.manage_trade("aggressive", date, idx)

    def manage_trade(self, trade_type, date, idx):
        if len(self.current_trade[trade_type]) != 0:
            # Step 1: Get the alert data and processed data for the ongoing trade
            stock = self.current_trade[trade_type]["symbol"]

            tracked_processed_stock = self.df[(self.df['symbol'] == stock) & (self.df['date'] == date)]
            tracked_alert_stock = self.alert_df[(self.alert_df['symbol'] == stock) &
                                                (self.alert_df['date'] == date) &
                                                (self.alert_df['interval'] == 1)]

            if tracked_alert_stock.empty:
                return

            # Step 2 : Aggressive selling rule: Sell if the stock fails to maintain velocity or inverse hammer
            if trade_type == "aggressive":
                self.protected = False

                # Initial protection flag and profit tracking
                if not self.dynamic_protection:
                    # Track current profit or loss for the stock
                    tracked_profit_loss = self.track_profit_loss(trade_type, tracked_processed_stock, sell=False)
                    tracked_profit_pct = (tracked_profit_loss - self.aggressive_capital) / self.aggressive_capital
                    # Activate dynamic protection if profit reaches or exceeds 10%
                    if tracked_profit_pct >= 0.3:
                        print(
                            f"{stock} in {tracked_processed_stock['date'].iloc[0]} needs attention in profit protection | current profit rate: {tracked_profit_pct}")
                        self.dynamic_protection = True
                        self.peak_profit = tracked_profit_loss  # Set the initial peak profit
                        self.peak_profit_pct = tracked_profit_pct
                    else:
                        self.dynamic_protection = False

                # Dynamic protection logic if the flag is activated
                elif self.dynamic_protection:
                    # Track updated profit or loss
                    tracked_profit_loss = self.track_profit_loss(trade_type, tracked_alert_stock, sell=False)
                    tracked_profit_pct = (tracked_profit_loss - self.aggressive_capital) / self.aggressive_capital

                    print(
                        f"{tracked_processed_stock['symbol'].iloc[0]}| {tracked_processed_stock['date'].iloc[0]}: current profit: {tracked_profit_loss} | current profit rate: {tracked_profit_pct}")

                    # Update the peak profit if the profit is increasing
                    if tracked_profit_loss > self.peak_profit:
                        self.peak_profit = tracked_profit_loss
                        self.peak_profit_pct = tracked_profit_pct
                        print(
                            f"New peak profit updated: {tracked_processed_stock['symbol'].iloc[0]} earns {self.peak_profit}")

                    # Sell if the profit has declined by 50% from the peak profit
                    if self.peak_profit_pct - tracked_profit_pct >= self.peak_profit_pct * 0.5:
                        print(
                            f"{stock} Profit declined by 50% from the peak. Initiating sell.| Peak profit rate: {self.peak_profit_pct} vs. Current profit rate: {tracked_profit_pct}")
                        self.track_profit_loss(trade_type, tracked_alert_stock, sell=True)
                        self.dynamic_protection = False  # Reset dynamic protection after selling
                        self.protected = True

                if not self.protected:
                    # Sell if bearish signals
                    if tracked_alert_stock['velocity_alert'].iloc[0] == 'velocity_loss':
                        self.track_profit_loss(trade_type, tracked_processed_stock, sell=True)

                    # Sell if testing duration ends
                    if idx == len(self.df['date'].unique()) - 1:
                        self.track_profit_loss(trade_type, tracked_alert_stock, sell=True)
                else:
                    return

        elif len(self.current_trade[trade_type]) == 0:
            # Step 1: Get the stock candidate for the day
            cur_stock_pick = self.stock_candidates.iloc[idx]
            # Step 2: Find the stock based on the desired alert
            stock = self.find_alert(trade_type, cur_stock_pick, desired_alerts=['accelerating'])
            # Step 3: Buy the stock if found
            if not stock:
                return
            # Step 3.1: Aggressive trade: Test buy with 10% of aggressive capital (Conservative trade will buy full)
            # if trade_type == "aggressive":
            #     self.current_trade[trade_type]["testing_buy"] = True  # Flag for testing buy
            #     self.current_trade[trade_type]["testing_capital"] = self.aggressive_capital * 0.1  # 10% of aggressive capital
            #     self.aggressive_capital *= 0.9  # Hold back 90% for potential all-in later

            # Step 4: Update the current trade
            entry_date = self.df['date'].iloc[idx + 1]
            self.current_trade[trade_type]["entry_price"] = \
                self.df[(self.df['symbol'] == stock) & (self.df['date'] == entry_date)]['open'].iloc[0]
            self.current_trade[trade_type]["entry_date"] = \
                self.df[(self.df['symbol'] == stock) & (self.df['date'] == entry_date)]['date'].iloc[0]
            self.current_trade[trade_type]["symbol"] = stock

    # Separate method to handle selling a stock
    def track_profit_loss(self, trade_type, tracked_processed_stock, sell=False):
        # Step 1: Calculate profit/loss rate
        exit_price = tracked_processed_stock['close'].iloc[0]
        entry_price = self.current_trade[trade_type]['entry_price']
        profit_rate = (exit_price / entry_price) - 1
        if sell:
            # Step 2: Update the capital based on the profit rate and trade type
            if trade_type == "aggressive":
                # # Check if it's a testing buy
                # if self.current_trade[trade_type]['testing_buy'] == True:
                #     # Apply profit rate only to testing buy capital if testing buy
                #     testing_capital = self.current_trade[trade_type]["testing_capital"]
                #     self.aggressive_capital += (testing_capital * (profit_rate + 1))
                # else:
                # Apply profit to full aggressive capital if it's fully invested
                self.aggressive_capital += self.aggressive_capital * profit_rate

            # Save the trade
            self.trades.append({
                "type": trade_type,
                "symbol": self.current_trade[trade_type]["symbol"],
                "Entry_price": entry_price,
                "Entry_date": self.current_trade[trade_type]["entry_date"],
                "Exit_price": exit_price,
                "Exit_date": tracked_processed_stock['date'].iloc[0],
                "profit/loss": f"{profit_rate * 100 :.2f}%",
                "total_conser_asset": self.conservative_capital,
                "total_aggr_asset": self.aggressive_capital,
                "total_asset": self.conservative_capital + self.aggressive_capital
            })

            # Reset current trade
            self.current_trade[trade_type] = {}
        else:
            if trade_type == "conservative":
                return self.conservative_capital + self.conservative_capital * profit_rate
            elif trade_type == "aggressive":
                return self.aggressive_capital + self.aggressive_capital * profit_rate

    def find_alert(self, trade_type, stock_data, desired_alerts: list):
        if trade_type == "aggressive":
            for alert in desired_alerts:
                cur_alert_data = stock_data.loc[alert]
                # Ensure that cur_stock_pick is non-empty and the alert column exists
                if len(cur_alert_data) != 0:
                    return stock_data[alert][0]

    def run_trading_strategy(self):
        self.execute_critical_trades()

    def get_trades(self):
        return pd.DataFrame(self.trades)

    def get_total_return(self):
        total_capital = self.conservative_capital + self.aggressive_capital
        return total_capital