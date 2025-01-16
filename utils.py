import numpy as np
import pandas as pd
from pymongo import MongoClient

class StrategyEDA:
    def __init__(self, mongo_config, start_date, end_date, buy_signals, sell_signals, instrument):
        self.mongo_config = mongo_config
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.instrument = instrument
        self.title = self.process_title(buy_signals, sell_signals)
    
    def process_title(self, buy_signals, sell_signals):
        buy_signals = ', '.join(buy_signals)
        sell_signals = ', '.join(sell_signals)
        return f'{self.instrument} Buy: {buy_signals} | Sell: {sell_signals}'
    
    def get_nasdaq_return_data(self):
        mongo_client = MongoClient(self.mongo_config['url'])
        nasdaq_data = pd.DataFrame(list(mongo_client[self.mongo_config['db_name']][self.mongo_config['warehouse_interval'] + '_data'].\
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
        mongo_client = MongoClient(self.mongo_config['url'])
        bitcoin_data = pd.DataFrame(list(mongo_client[self.mongo_config['db_name']][self.mongo_config['warehouse_interval'] + '_data'].\
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
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Convert profit/loss from string to float
        results_df['profit_loss_float'] = results_df['profit/loss'].str.rstrip('%').astype(float)
        
        # Create subplots with different heights
        fig = make_subplots(rows=2, cols=1, 
                        row_heights=[0.7, 0.3],
                        subplot_titles=(f'Portfolio Value Over Time',
                                        f'Distribution of Returns'))
        
        # Process data for portfolio value plot
        results_df_sorted = results_df.sort_values('Exit_date')
        first_data = pd.DataFrame({'total_captial': [10000], 'Exit_date': [pd.to_datetime(self.start_date)]})
        pd.set_option("future.no_silent_downcasting", True)
        results_df_sorted = pd.concat([first_data, results_df_sorted]).fillna(0)
        results_df_sorted = results_df_sorted.reset_index(drop=True)
        results_df_sorted.set_index('Exit_date', inplace=True)

        # Create date range and interpolate
        date_range = pd.date_range(start=pd.to_datetime(self.start_date), 
                                end=pd.to_datetime(self.end_date), 
                                freq='D')
        results_df_sorted = results_df_sorted.reindex(date_range)
        results_df_sorted['total_captial'] = results_df_sorted['total_captial'].astype(float)
        results_df_sorted['total_captial'] = results_df_sorted['total_captial'].interpolate()
        results_df_sorted['monthly_return'] = results_df_sorted['total_captial'].pct_change(periods=30)
        
        # Add Comparison Data
        if self.instrument == 'equity':
            nasdaq_data = self.get_nasdaq_return_data()
        elif self.instrument == 'crypto':
            bitcoin_data = self.get_bitcoin_return_data()
        else:
            raise ValueError(f"Invalid instrument: {self.instrument}")
        
        # Plot strategy line
        fig.add_trace(
            go.Scatter(x=results_df_sorted.index, 
                    y=results_df_sorted['total_captial'],
                    name='Strategy',
                    line=dict(color='blue')),
            row=1, col=1
        )
        
        # Plot Bitcoin
        if self.instrument == 'crypto':
            fig.add_trace(
                go.Scatter(x=bitcoin_data.index, 
                        y=bitcoin_data['total_captial'],
                    name='Bitcoin',
                    line=dict(color='rgba(247, 147, 26, 0.5)')),  # Bitcoin orange
            row=1, col=1
            )
        # Plot NASDAQ
        if self.instrument == 'equity':
            fig.add_trace(
                go.Scatter(x=nasdaq_data.index, 
                    y=nasdaq_data['total_captial'],
                    name='NASDAQ',
                    line=dict(color='rgba(85, 85, 85, 0.5)')),  # Dark gray
            row=1, col=1
        )
        
        # Add initial principle line
        fig.add_hline(y=10000, line_dash="dash", line_color="gray", 
                    annotation_text="Initial Principle",
                    row=1, col=1)
    
        # Add bear market shading
        fig.add_vrect(
            x0="2022-01-01", x1="2022-12-31",
            fillcolor="red", opacity=0.1,
            layer="below", line_width=0,
            row=1, col=1
        )
        
        # Add final values annotation
        strategy_final = results_df_sorted['total_captial'].iloc[-1]
        if self.instrument == 'equity':
            nasdaq_final = nasdaq_data['total_captial'].iloc[-1]
        elif self.instrument == 'crypto':
            bitcoin_final = bitcoin_data['total_captial'].iloc[-1]
        else:
            raise ValueError(f"Invalid instrument: {self.instrument}")
        
        # Position annotations based on log scale
        max_y = results_df_sorted['total_captial'].max()
        
        if self.instrument == 'equity':
            fig.add_annotation(
                text=f'Strategy Final: ${strategy_final:,.2f} (NASDAQ Final: ${nasdaq_final:,.2f})',
                x=pd.to_datetime(self.start_date) + pd.Timedelta(days=365),
                y=max_y * 0.9,
                showarrow=False,
                row=1, col=1
            )
        elif self.instrument == 'crypto':
            fig.add_annotation(
                text=f'Strategy Final: ${strategy_final:,.2f} (Bitcoin Final: ${bitcoin_final:,.2f})',
                x=pd.to_datetime(self.start_date) + pd.Timedelta(days=365),
                y=max_y * 0.9,
                showarrow=False,
                row=1, col=1
            )
        else:
            raise ValueError(f"Invalid instrument: {self.instrument}")
        # Add monthly return annotations adjusted for log scale
        for i in range(0, len(results_df_sorted), 30):
            if i+30 < len(results_df_sorted):
                monthly_return = results_df_sorted['monthly_return'].iloc[i+ 30]
                if not pd.isna(monthly_return):
                    date = results_df_sorted.index[i+ 30]
                    current_value = results_df_sorted['total_captial'].iloc[i+ 30]
                    y_pos = current_value * 1.1  # Position 10% above the current value 
                    color = 'green' if monthly_return > 0 else 'red' 
                    fig.add_annotation( 
                        text=f'{monthly_return:.3%}', 
                        x=date, 
                        y=y_pos, 
                        showarrow=False, 
                        font=dict(color=color, size=10), 
                        textangle=90, 
                        row=1,  
                        col=1
                    )
        
        # Add returns distribution plot
        fig.add_trace(
            go.Histogram(x=results_df['profit_loss_float'],
                        name='Returns Distribution',
                        nbinsx=180),
            row=2, col=1
        )
        
        # Add vertical line at x=0 for distribution
        fig.add_vline(x=0, line_dash="dash", line_color="red", row=2, col=1)

        # Update layout with log y-axis
        fig.update_layout(
            height=800,
            width=1200,
            showlegend=True,
            title_text=self.title,
            xaxis=dict(
                range=[pd.to_datetime(self.start_date), pd.to_datetime(self.end_date)]
            ),
            yaxis=dict(type='linear')  # Set y-axis to logarithmic scale
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Return %", row=2, col=1,)
        fig.update_yaxes(title_text="Total Amount ($)", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        
        # Show the plot
        fig.show()
        
