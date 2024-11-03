import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class DailyTradingStrategy:
    def __init__(self, data_collection, alert_collection, stock_candidates,
                 start_date=None, end_date=None, sandbox_mode=False,
                 initial_capital=10000, aggressive_split=1.):

        self.protected = None
        self.peak_profit_pct = None
        self.peak_profit = None
        self.trades = []
        self.current_trade = {"conservative": {}, "aggressive": {}}
        self.start_date = start_date if sandbox_mode else '2024-01-01'
        self.end_date = end_date if sandbox_mode else '2024-10-25'
        # Define split for aggressive and conservative capital
        self.aggressive_capital = initial_capital * aggressive_split
        self.conservative_capital = initial_capital * (1.0 - aggressive_split)
        self.capital_split = aggressive_split  # Correct spelling
        self.data_collection = data_collection
        self.alert_collection = alert_collection
        self.dynamic_protection = False
        self.df = pd.DataFrame(list(self.data_collection.find({'date': {
            '$gte': pd.to_datetime(self.start_date),
            "$lte": pd.to_datetime(self.end_date)},
            'interval': 1
        },
            {'_id': 0}))) \
            .sort_values(by=['symbol', 'date'])

        self.alert_df = self.get_alert_dataframe()
        self.stock_candidates = stock_candidates[
            pd.to_datetime(stock_candidates['date']) >= pd.to_datetime(self.start_date)]
        self.aggressive_hold_day = 0

    def get_alert_dataframe(self):

        # Fetch the data from MongoDB and convert to DataFrame
        data_dict = list(self.alert_collection.find({'date': {'$gte': pd.to_datetime(self.start_date),
                                                              "$lte": pd.to_datetime(self.end_date)}},
                                                    {'_id': 0}))

        # Extract the alerts from the alert_dict
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

        # Convert the alert_dict to a DataFrame
        data = pd.DataFrame(data_dict)
        data = data.drop(columns=['alerts'])

        return data

    # Execute both types of trades
    def execute_critical_trades(self):

        for idx, date in enumerate(self.df['date'].unique()):
            # Handle aggressive trades
            if self.aggressive_capital >= 0:
                self.manage_trade("aggressive", date, idx)

    def manage_trade(self, trade_type, date, idx):

        # ==================== Selling Logic =========================== #
        # If there is an ongoing trade for the day                       #
        # Check if the stock is still a good hold based on the alerts    #
        # ============================================================== #
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

                # ================== Aggressive Capital Protection ================== #
                # If the total capital falls below the initial capital, stop trading  #
                # =================================================================== #

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

                    # Update the peak profit if the profit is increasing
                    if tracked_profit_loss > self.peak_profit:
                        self.peak_profit = tracked_profit_loss
                        self.peak_profit_pct = tracked_profit_pct

                    # Sell if the profit has declined by 50% from the peak profit
                    if self.peak_profit_pct - tracked_profit_pct >= self.peak_profit_pct * 0.5:

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

        # ==================== Buying Logic =========================== #
        # If there is no ongoing trade for the day                      #
        # Check if there is a stock candidate for the day               #
        # ============================================================= #

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

    # def chase_trade(self, trade_type, date):
    #     stock = self.current_trade[trade_type]["symbol"]
    #     tracked_processed_stock = self.df[(self.df['symbol'] == stock) & (self.df['date'] == date)]

    #     # Apply the remaining capital to the aggressive trade
    #     self.aggressive_capital += self.current_trade[trade_type]["testing_capital"]
    #     self.current_trade[trade_type]["testing_capital"] = 0

    #     # Compute the average entry price
    #     self.current_trade[trade_type]["entry_price"] = (self.current_trade[trade_type]["entry_price"] + tracked_processed_stock['close'].iloc[0]) / 2
    #     self.current_trade[trade_type]["entry_date"] = tracked_processed_stock['date'].iloc[0]
    #     self.current_trade[trade_type]["symbol"] = stock
    #     self.current_trade[trade_type]["testing_buy"] = False  # Reset testing buy flag

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

class PickStock:

    def __init__(self, alert_collection, start_date=None, sandbox_mode=False):
        # Set Sandbox mode and start date accordingly
        self.sandbox_mode = sandbox_mode
        if self.sandbox_mode:
            self.start_date = start_date
        else:
            self.start_date = '2024-01-01'
        # Database collection name
        self.alert_collection = alert_collection

        # Initialize a one hot encoder
        self.ohe = OneHotEncoder(sparse_output=False)

        # Fetch the data, sorted by symbol and date
        self.data = self.get_stock_dataframe()
        self.distinct_intervals = self.data['interval'].unique()

        # Dictionary to store stock candidates
        self.stock_candidates = {}

        # Define weights for intervals (e.g., larger intervals get more weight)
        self.interval_weights = {interval: weight for weight, interval in enumerate(self.distinct_intervals, start=1)}

    def get_stock_dataframe(self):
        # Fetch the data from MongoDB and convert to DataFrame
        alert_dict = list(self.alert_collection.find({'date': {'$gte': pd.to_datetime(self.start_date)}}, {'_id': 0}))
        # Extract the alerts from the alert_dict
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

        # Convert the alert_dict to a DataFrame
        data = pd.DataFrame(alert_dict)
        data = data.drop(columns=['alerts'])

        # Prepare the data for encoding
        alert_columns = ['touch_type', 'momentum_alert']
        df_prep_encoded = data.loc[:, alert_columns]
        data.drop(columns=alert_columns, inplace=True)

        # Encode the data
        encoded_arr = self.ohe.fit_transform(df_prep_encoded)
        encoded_df = pd.DataFrame(encoded_arr, columns=self.ohe.get_feature_names_out())

        # Concat the encoded dataframe to the alert dataframe
        encoded_alert_data = pd.concat([data, encoded_df], axis=1)

        return encoded_alert_data

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
        acc_equ = results[(results['weighted_momentum_alert_accelerated'] > 0) \
                          & (results['weighted_momentum_alert_decelerated'] < 1) \
                          & (results['interval'] <= 3)].loc[:, 'symbol']

        # Store the main force accumulating stock
        main_acc_equ = results[(results['weighted_touch_type_support'] > 0) \
                               & (results['weighted_touch_type_resistance']) < 1 \
                               & (results['count'] > 2) \
                               & (results['interval'] <= 3)].loc[:, 'symbol']

        # Store the stable stock in high interval
        stable = results[(results['weighted_touch_type_support'] > 0) \
                         & (results['weighted_touch_type_resistance']) < 1 \
                         & (results['count'] > 1) \
                         & (results['interval'] >= 3)].loc[:, 'symbol']

        # Create dictionary of results
        stock_dict = {
            'accelerating': acc_equ.tolist(),
            'main_accumulating': main_acc_equ.tolist(),
            'stable': stable.tolist()
        }

        return stock_dict

    def run(self):
        # Initialize a dictionary to hold the results with dates as keys
        stock_candidates = {}

        for date in self.data['date'].unique():
            # Process and analyze the candidates stock of the day
            today_data = self.data[self.data['date'] == date]
            today_candidates = self.evaluate_stocks(today_data)

            # Assign today's candidates to the dictionary with date as the key
            stock_candidates[str(date)] = today_candidates  # Ensure date is string for JSON compatibility

        # Create a DataFrame with date and the symbol categories
        candidates_df = pd.DataFrame([
            {'date': date, 'accelerating': candidates['accelerating'],
             'main_accumulating': candidates['main_accumulating'],
             'stable': candidates['stable']}
            for date, candidates in stock_candidates.items()
        ])

        return candidates_df
