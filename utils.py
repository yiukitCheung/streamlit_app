import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from pymongo import MongoClient
URL = st.secrets['mongo']['host']
DB_NAME = st.secrets['mongo']['db_name']
WAREHOUSE_INTERVAL = st.secrets['mongo']['warehouse_interval']
# Plot return chart
class StrategyEDA:
    def __init__(self, start_date, end_date, instrument="equity"):
        self.start_date = start_date
        self.end_date = end_date
        self.instrument = instrument
        
    def get_nasdaq_return_data(self):
            mongo_client = MongoClient(URL)
            nasdaq_data = pd.DataFrame(list(mongo_client[DB_NAME][WAREHOUSE_INTERVAL + '_data'].\
                find({'symbol': '^IXIC',
                        'date': {'$gte': self.start_date, '$lte': self.end_date},
                        },
                        {'date': 1, 'close': 1, '_id': 0})))
            nasdaq_data['return'] = nasdaq_data['close'].pct_change()
            nasdaq_data.dropna(inplace=True)
            
            # Assuming 10000 initial capital
            nasdaq_data['total_captial'] = 10000 * (1 + nasdaq_data['return']).cumprod()
            return nasdaq_data
        
    def get_bitcoin_return_data(self):
        mongo_client = MongoClient(URL)
        bitcoin_data = pd.DataFrame(list(mongo_client[DB_NAME][WAREHOUSE_INTERVAL + '_data'].\
            find({'symbol': 'BTC',
                    'date': {'$gte': self.start_date, '$lte': self.end_date},
                    },
                    {'date': 1, 'close': 1, '_id': 0})))
        bitcoin_data['return'] = bitcoin_data['close'].pct_change()
        bitcoin_data.dropna(inplace=True)
            
        # Assuming 10000 initial capital
        bitcoin_data['total_captial'] = 10000 * (1 + bitcoin_data['return']).cumprod()    
        return bitcoin_data
    
    def plot_trading_analysis(self, results_df):
        # Convert profit/loss from string to float
        results_df = pd.DataFrame(results_df)
        results_df = results_df[results_df['instrument'] == self.instrument]
        # Convert profit/loss to floattotal_captial
        results_df['profit_loss_float'] = results_df['profit/loss%'].str.rstrip('%').astype(float) / 100
        results_df['good_trade'] = results_df['profit_loss_float'] > 0.1
        results_df['total_captial'] = results_df['total_captial'].astype(float)
        # Create subplots with different heights
        fig = make_subplots(rows=1, cols=1)
        
        # Process data for portfolio value plot
        results_df_sorted = results_df.sort_values('Exit_date')
        first_data = pd.DataFrame({'total_captial': [10000], 'Exit_date': [pd.to_datetime(self.start_date)]})
        results_df_sorted = pd.concat([first_data, results_df_sorted]).fillna(0)
        results_df_sorted = results_df_sorted.reset_index(drop=True)
        
        # Ensure Exit_date is unique before setting as index
        results_df_sorted = results_df_sorted.groupby('Exit_date').last().reset_index()
        results_df_sorted.set_index('Exit_date', inplace=True)

        # Create date range and interpolate
        date_range = pd.date_range(start=pd.to_datetime(self.start_date), 
                                end=pd.to_datetime(self.end_date), 
                                freq='D')
        
        results_df_sorted = results_df_sorted.reindex(date_range)
        results_df_sorted['total_captial'] = results_df_sorted['total_captial'].interpolate()
        # Add Comparison Data
        if self.instrument == 'equity':
            nasdaq_data = self.get_nasdaq_return_data()
        elif self.instrument == 'crypto':
            bitcoin_data = self.get_bitcoin_return_data()
        else:
            raise ValueError(f"Invalid instrument: {self.instrument}")
        
        # Calculate returns for annotations
        strategy_return = ((float(results_df_sorted['total_captial'].iloc[-1]) / float(results_df_sorted['total_captial'].iloc[0])) - 1) * 100
        # Plot strategy line
        fig.add_trace(
            go.Scatter(x=results_df_sorted.index, 
                    y=results_df_sorted['total_captial'],
                    name='Strategy',
                    showlegend=False,
                    line=dict(color='#2E8B57', width=3)),
            row=1, col=1
        )

        # Add markers and text for good trades
        good_trades = results_df[results_df['good_trade']]
        for _, trade in good_trades.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[pd.to_datetime(trade['Exit_date'])],
                    y=[trade['total_captial']],
                    mode='markers+text',
                    marker=dict(symbol='star', size=25, color='gold'),
                    text=f"<b>{trade['symbol']}</b>",
                    textposition='top center',
                    showlegend=False,
                    textfont=dict(size=25, color='rgba(46, 139, 87, 0.7)')
                ),
                row=1, col=1
            )
        
        # Plot Bitcoin or NASDAQ with annotations
        if self.instrument == 'crypto':
            bitcoin_return = ((bitcoin_data['total_captial'].iloc[-1] / bitcoin_data['total_captial'].iloc[0]) - 1) * 100
            fig.add_trace(
                go.Scatter(x=bitcoin_data.index, 
                        y=bitcoin_data['total_captial'],
                        showlegend=False,
                        line=dict(color='rgba(247, 147, 26, 0.5)', width=2)),
                row=1, col=1
            )
            # Add cumulative return text with enhanced styling and comparison
            outperformance = strategy_return - bitcoin_return
            performance_color = '#2E8B57' if outperformance > 0 else '#DC143C'
            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=f'<b> Crypto Performance Analysis</b><br><br>' +
                    f'<span style="color:{performance_color}">CondVest: <b>{strategy_return:+.1f}%</b></span><br>' +
                    f'Bitcoin: {bitcoin_return:+.1f}%<br><br>' +
                    f'<span style="color:{performance_color}">Outperformance: <b>{outperformance:+.1f}%</b></span>',
                showarrow=False,
                align="left",
                font=dict(size=18)
            )
        elif self.instrument == 'equity':
            nasdaq_return = ((nasdaq_data['total_captial'].iloc[-1] / nasdaq_data['total_captial'].iloc[0]) - 1) * 100
            fig.add_trace(
                go.Scatter(x=nasdaq_data.index, 
                        y=nasdaq_data['total_captial'],
                        showlegend=False,
                        line=dict(color='rgba(85, 85, 85, 0.5)', width=2)),
                row=1, col=1
            )
            # Add cumulative return text with enhanced styling and comparison
            outperformance = strategy_return - nasdaq_return
            performance_color = '#2E8B57' if outperformance > 0 else '#DC143C'
            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=f'<b> Stock Performance Analysis</b><br>' +
                    f'<span style="color:{performance_color}">CondVest: <b>{strategy_return:+.1f}%</b></span><br>' +
                    f'NASDAQ: {nasdaq_return:+.1f}%<br>' +
                    f'<span style="color:{performance_color}">Outperformance: <b>{outperformance:+.1f}%</b></span>',
                showarrow=False,
                align="left",
                font=dict(size=18)
                
            )

        # Update layout with log y-axis and prominent axes
        fig.update_layout(
            height=650,
            showlegend=False,
            title=dict(
                text="CondVest Empowers You to Win the Market !!!",
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=24, color='#2E8B57')
            ),
            xaxis=dict(
                range=[pd.to_datetime(self.start_date), pd.to_datetime(self.end_date)],
                showgrid=False,
                mirror=False,
                tickfont=dict(size=16, color='black', family='Arial Bold')  # Made tick values more prominent
            ),
            yaxis=dict(
                type='log',
                showgrid=False,
                mirror=False,
                tickfont=dict(size=16, color='#2E8B57', family='Arial Bold')  # Made tick values more prominent
            )
        )
        
        # Update axes labels to match overall style
        fig.update_xaxes(
            title_text="Date", 
            title_font=dict(size=14, color='#2E8B57', family='Arial'),
            row=1, 
            col=1
        )
        fig.update_yaxes(
            title_text="Total Amount ($)", 
            title_font=dict(size=14, color='#2E8B57', family='Arial'),
            row=1, 
            col=1
        )

        return fig
