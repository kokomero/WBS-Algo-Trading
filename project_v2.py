import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd
import datetime as dt
import pathlib
import time

import market_data as md
import alpha_vantage_api as av
import estimators
import performance_estimators as pe
import strategies as strat

# Read dataset
readFromFile: bool = True
plotGraphs: bool = False

if readFromFile:
    # CSV with SOFR
    sofr = md.load_sofr_dataset(file_path=r'SOFR.csv')
    stocks = md.load_spot_dataset(file_path=r'yahoo_data.csv')
    raw_data = stocks.merge(sofr, on='timestamp', how='inner')
else:
    # prices = pd.read_csv( r'prices.csv',
    #                       sep=';',
    #                       index_col='timestamp',
    #                       parse_dates = True)
    pass

etf_list = ['SPY', 'QQQ', 'SOXX', 'VOO', 'IWM', 'SQQQ', 'SOXS',
            'XLE', 'XLK', 'XLV', 'IVV', 'INDA', 'ARKK', 'EEM',
            'FXI', 'SXRT', 'TLT', 'LQD', 'IEF', 'SHY', 'VIXY']
#underlying = etf_list[4]
underlying = 'SPY'
fields = ['Adj Close']
init_time = dt.date(2010, 1, 1)

prices_df = md.get_historical_prices(prices=raw_data,
                                     underlying=underlying,
                                     fields=fields,
                                     funding_name='sofr')\
    .rename(columns={'Adj Close': 'close'})\
    .loc[lambda df: df.index.date > init_time]

# Define Parameters for trading strategy
sma_params = strat.SMA_Params( 
    execution_fee = 2.0 / 10_000,
    mean_slippage = 1.0/10_000,
    borrowing_cost = 3.0 / 100.0,
    bet_size = 10_000,  # in cash term,
    # vol_scaling_func = None,
    vol_scaling_func = lambda vol: np.min([1.0,0.15/vol]),
    short_sma_window = 10,
    long_sma_window = 50,
    feature_col = 'sma_long'
    )

# Define parameters for performance statistics
perf_params = {
    'var_pctl': 0.99,
    'risk_free': 0.0,
    'target_return': 0.0
}

# Calculate features

# Primero dado un ticker, selecciona historico de ese ticker y transforma la serie para que la entienda run_strategy

# Explore later on open, high, low returns
# Equities settle at T+2, account for that
strategy_df = strat.run_sma_strategy(prices=prices_df,
                                     params=sma_params)

# Calculate performance metrics
benchmark_perf = pe.calc_performance(df=strategy_df,
                                     return_field='close_ret',
                                     params=perf_params,
                                     price_rets='price')
strategy_perf = pe.calc_performance(df=strategy_df,
                                    return_field='strat_ret',
                                    params=perf_params,
                                    price_rets='strategy_returns')
benchmark_perf_df = pe.performance_df(benchmark_perf, f"{underlying}")
strategy_perf_df = pe.performance_df(strategy_perf, f"SMA on {underlying}")
perf_df = pd.concat([benchmark_perf_df, strategy_perf_df], axis=1)
print(perf_df)



if plotGraphs:
    # Create plot for strategy
    signal_fig = go.Figure()

    # Underlying plot
    benchmark_plt = go.Scatter(x=strategy_df.index,
                            y=strategy_df.close,
                            name='Close',
                            line=dict(color='firebrick', width=1))

    # Feature (SMA) plot
    feature_plt = go.Scatter(x=strategy_df.index,
                            y=strategy_df[sma_params.feature_col],
                            name=sma_params.feature_col,
                            line=dict(color='royalblue', width=1, dash='dot'))

    # Sell/Buy signal plot
    # Identify position changes
    short_idx = (strategy_df.signal == -1.0)
    long_idx = (strategy_df.signal == 1.0)
    cross_below_plt = go.Scatter(x=strategy_df[short_idx].index,
                                y=strategy_df[short_idx].sma_long,
                                name='Sell signal',
                                mode='markers',
                                marker_symbol='triangle-down',
                                marker=dict(color='red', size=8))
    cross_above_plt = go.Scatter(x=strategy_df[long_idx].index,
                                y=strategy_df[long_idx].sma_long,
                                name='Buy Signal',
                                mode='markers',
                                marker_symbol='triangle-up',
                                marker=dict(color='green', size=8))

    signal_fig.add_trace(benchmark_plt)
    signal_fig.add_trace(feature_plt)
    signal_fig.add_trace(cross_below_plt)
    signal_fig.add_trace(cross_above_plt)
    signal_fig.update_yaxes(title_text="Prices")
    signal_fig.update_layout(title={
        'text': f'Strategy on {underlying}, using {sma_params.feature_col} feature and Generated signals',
        'x': 0.5})
    signal_fig.show()

    # Histogram of strategy returns vs underlying returns
    returns_histogram_fig = go.Figure()
    bins_cfg = dict(  # bins used for histogram
        start=-0.06,
        end=0.06,
        size=0.0025
    )
    strategy_ret_plt = go.Histogram(x=strategy_df.strat_ret,
                                    name='Strategy returns',
                                    xbins=bins_cfg)
    underlying_ret_plt = go.Histogram(x=strategy_df.close_ret,
                                    name='Underlying returns',
                                    xbins=bins_cfg)
    returns_histogram_fig.add_trace(strategy_ret_plt)
    returns_histogram_fig.add_trace(underlying_ret_plt)
    returns_histogram_fig.update_yaxes(title_text="Daily Occurences")
    returns_histogram_fig.update_xaxes(title_text="Daily Returns")
    returns_histogram_fig.update_layout(title={
        'text': f'Daily Returns Histogram',
        'x': 0.5},
        bargap=0.2,  # gap between bars of adjacent location coordinates
        bargroupgap=0.1,  # gap between bars of the same location coordinates)
        xaxis=dict(tickformat="0.2%")
    )
    returns_histogram_fig.update_traces(opacity=0.75)
    returns_histogram_fig.show()

    # Comparative accumulated performance between underlying and strategy
    comp_perf_fig = go.Figure()
    strategy_perf_plt = go.Scatter(x=strategy_df.index,
                                y=strategy_df.strat_pnl /
                                strategy_df.strat_pnl.iloc[0],
                                mode='lines',
                                name=f'Strategy',
                                line=dict(
                                    color='firebrick',
                                    width=1)
                                )
    underlying_perf_plt = go.Scatter(x=strategy_df.index,
                                    y=strategy_df.close/strategy_df.close.iloc[0],
                                    mode='lines',
                                    name=f'Underlying {underlying}',
                                    line=dict(
                                        color='royalblue',
                                        width=1)
                                    )
    comp_perf_fig.add_trace(strategy_perf_plt)
    comp_perf_fig.add_trace(underlying_perf_plt)
    comp_perf_fig.update_yaxes(title_text='Comparative Performance')
    comp_perf_fig.update_layout(title={
        'text': f'Comparative performance between {underlying} and {sma_params.feature_col} strategy',
        'x': 0.5})
    comp_perf_fig.show()


    # Compare daily volatility vs returns
    vol_fig_fig = go.Figure()
    strategy_vol_plt = go.Scatter(x=strategy_df.strategy_vol,
                                y=strategy_df.strat_ret,
                                mode='markers',
                                marker_symbol='x',
                                name=f'Strategy',
                                marker=dict(
                                    color='firebrick',
                                    size=3)
                                )
    benchmark_vol_plt = go.Scatter(x=strategy_df.benchmark_vol,
                                y=strategy_df.close_ret,
                                mode='markers',
                                marker_symbol='circle',
                                name=f'Underlying {underlying}',
                                marker=dict(
                                    color='royalblue',
                                    size=3)
                                )
    vol_fig_fig.add_trace(strategy_vol_plt)
    vol_fig_fig.add_trace(benchmark_vol_plt)
    vol_fig_fig.update_yaxes(title_text='Returns')
    vol_fig_fig.update_xaxes(title_text='EWMA Vol')
    vol_fig_fig.update_layout(title={
        'text': f'Daily Returns vs Volatility for {underlying} and {sma_params.feature_col} strategy',
        'x': 0.5},
        xaxis=dict(tickformat="0.2%", range=[0.0, 0.4]),
        yaxis=dict(tickformat="0.2%", range=[-0.04, +0.04]))
    vol_fig_fig.show()

    # Plot benchmark volatility and where the trades are done
    vol_trades_fig = go.Figure()
    bench_vol_ts_plt = go.Scatter(x=strategy_df.index,
                                y=strategy_df.benchmark_vol,
                                mode='lines',
                                name=f'EWMA Vol for {underlying}',
                                line=dict(
                                        color='royalblue',
                                        width=1)
                                )

    cross_above_plt = go.Scatter(x=strategy_df[long_idx].index,
                                y=strategy_df[long_idx].benchmark_vol,
                                name='Buy Signal',
                                mode='markers',
                                marker_symbol='triangle-up',
                                marker=dict(color='green', size=8))

    cross_below_plt = go.Scatter(x=strategy_df[short_idx].index,
                                y=strategy_df[short_idx].benchmark_vol,
                                name='Sell signal',
                                mode='markers',
                                marker_symbol='triangle-down',
                                marker=dict(color='red', size=8))
    vol_trades_fig.add_trace(bench_vol_ts_plt)
    vol_trades_fig.add_trace(cross_above_plt)
    vol_trades_fig.add_trace(cross_below_plt)
    vol_trades_fig.update_yaxes(title_text='Vol Level')
    vol_trades_fig.update_layout(title={
        'text': f'Daily Volatility for {underlying} and trades done for {sma_params.feature_col}',
        'x': 0.5}
        )
    vol_trades_fig.show()

"""
The following section will explore the cross-over between two SMA time-series, one
with short and the other with long windows length
"""
# Test the cross-over strategy between two SMA
# When short SMA is 1, this is equivalent to a simple SMA crossover
# against price. This is included for testing purposes
sma_cross_params = strat.SMA_Params( 
    execution_fee = 2.0 / 10_000,
    mean_slippage = 1.0/10_000,
    borrowing_cost = 3.0 / 100.0,
    bet_size = 10_000,  # in cash term,
    # vol_scaling_func = None,
    vol_scaling_func = lambda vol: np.min([1.0,0.15/vol]),
    short_sma_window = 1,
    long_sma_window = 50,
    feature_col = 'crossover'
    )

strategy_df = strat.run_sma_strategy(prices=prices_df,
                                     params=sma_cross_params)

# Calculate performance metrics
benchmark_perf = pe.calc_performance(df=strategy_df,
                                     return_field='close_ret',
                                     params=perf_params,
                                     price_rets='price')
strategy_perf = pe.calc_performance(df=strategy_df,
                                    return_field='strat_ret',
                                    params=perf_params,
                                    price_rets='strategy_returns')
benchmark_perf_df = pe.performance_df(benchmark_perf, f"{underlying}")
strategy_perf_df = pe.performance_df(strategy_perf, f"SMA on {underlying}")
perf_df = pd.concat([benchmark_perf_df, strategy_perf_df], axis=1)
print(perf_df)

"""
Training of the strategy to look for best parameter combination

First chose a training set which is 70% of the data-set, then
Apply the optimized parameters to the 30% remaning dataset,
running the strategy on only this portion of the dates
"""

#Define training and test set
training_size = 0.6
test_size = 1-training_size
data_set_size = prices_df.shape[0]
split_index = int(data_set_size*test_size)
training_df = prices_df.iloc[0:split_index]
test_df = prices_df.iloc[split_index:]

# Define SMA main strategy
sma_cross_params = strat.SMA_Params( 
            execution_fee = 2.0 / 10_000,
            mean_slippage = 1.0/10_000,
            borrowing_cost = 3.0 / 100.0,
            bet_size = 10_000,  # in cash term,
            vol_scaling_func = None,
            # vol_scaling_func = lambda vol: np.min([1.0,0.15/vol]),
            short_sma_window = 1,
            long_sma_window = 50,
            feature_col = 'crossover'
        )

#Grid search for best strategy
acc: int = 0
init_time: float = time.time()
perf_cache = {}
for w_short in range(2, 50, 5):
    for w_long in range(w_short+5,200,10):
        acc+=1

        sma_cross_params.short_sma_window = w_short
        sma_cross_params.long_sma_window = w_long
        strategy_df = strat.run_sma_strategy(prices=training_df,
                                             params=sma_cross_params)
            
        strategy_perf = pe.calc_performance(df=strategy_df,
                                    return_field='strat_ret',
                                    params=perf_params,
                                    price_rets='strategy_returns')
        
        perf_cache[(w_short, w_long)]=strategy_perf

        if acc % 50 == 0:
            lapse = time.time()
            calc_min = (acc/(lapse-init_time))*60
            print( f"Evaluating {sma_cross_params.short_sma_window}:{sma_cross_params.long_sma_window}, calc_min: {calc_min}")

for k,v in perf_cache.items():
    print(f"Parameters: {k}, Perf: {v['return'] + v['omega_ratio']/10}")

# Evaluation on test_set
strategies = [(k,v) for k,v in perf_cache.items()]
performances = [v['return'] + v['omega_ratio']/10 for k, v in strategies]
winner_strategy = strategies[np.argmax(np.array( performances))]

sma_cross_params.short_sma_window = winner_strategy[0][0]
sma_cross_params.long_sma_window = winner_strategy[0][1]
strategy_df = strat.run_sma_strategy(prices=test_df,
                                     params=sma_cross_params)
    
strategy_perf = pe.calc_performance(df=strategy_df,
                                    return_field='strat_ret',
                                    params=perf_params,
                                    price_rets='strategy_returns')


print( 'Stop here')