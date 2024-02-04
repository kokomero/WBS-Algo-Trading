"""Run a strategy given a dataset and generated signals."""
import numpy as np
import pandas as pd
import datetime as dt

from dataclasses import dataclass
from typing import Optional
from collections.abc import Callable

import estimators

@dataclass
class SMA_Params:
    """This class gathers the parameters for a SMA strategy
    """
    execution_fee: float # in relative terms
    mean_slippage: float # in relative terms
    borrowing_cost: float # in relative terms
    bet_size: float  # in cash terms
    short_sma_window: int # in number of days
    long_sma_window: int # in number of days
    feature_col: str # The column name of the data set to run the SMA on
    vol_scaling_func: Optional[ Callable[[float], float ]] = None #Vol Scaling function, takes vol estimation as input, returns bet size scaling within [0,1]    

def compound_cash(cashflows: pd.Series, daily_rate: pd.Series) -> pd.Series:
    """Take a stream of cashflow and compound the interest rates. 

    Args:
        cashflows (pd.Series): series of cashflows, the index of the 
        series is the timestamp of the cashflow
        daily_rate (pd.Series): daily rate of the interest rate
        used to compound the cash account

    Returns:
        pd.Series: cash account with the cash value for each
        index (timestamp)
    """
    # Bring together cashflow and rate series
    df = pd.concat(objs=[cashflows, daily_rate], axis=1)
    df.columns = ['cashflows', 'd_rate']

    # Get difference of between timestamps and compute the daily cash return
    df = df.assign(days=lambda df: df.index.diff().days.fillna(0))\
           .assign(interest=lambda x: x.d_rate*x.days/365)\
           .assign(compounded=0.0)

    # Set first element with the value of the first cashflow, if any
    df.loc[df.index[0], 'compounded'] = df.loc[df.index[0], 'cashflows']

    # Compund each day the cash position
    for n, n_1 in zip(df.index[1:], df.index[0:-1]):
        df.loc[n, 'compounded'] = (
            df.loc[n_1, 'compounded'])*(1+df.loc[n_1, 'interest'])+df.loc[n, 'cashflows']

    return pd.Series(df.compounded)

def signal_sma(df: pd.DataFrame, feature_col: str, trigger_col: str = 'close') -> pd.Series:
    """Generate buy/sell signals from a given feature.

    For the case of an SMA, feature column will be the value of the SMA and
    trigger columns will be, usually, closing prices, so that the execution
    of signal takes places at the end of the trading day when the crossing
    happens

    Args:
        df (pd.DataFrame): dataframe with price series as well as calculated
        SMA
        feature_col (str): name of column used to calculate the signal
        trigger_col (str, optional): name of columns triggering the buy/sell 
        actions. Defaults to 'close'.

    Returns:
        pd.Series: Series for each date in the data set with -1.0 if the 
        signal is to sell, +1.0 if the signal is to buy, and nan otherwise
    """
    # Get SMA and closing series, current and delayed series
    feature, trigger = df[feature_col], df[trigger_col]
    feature_d, trigger_d = feature.shift(-1), trigger.shift(-1)

    # When the previous day closing was lower than SMA but today's closing
    # is higher than SMA, cross below , i.e. the SMA cross 
    # the price trigger (close) from being above to being below
    idx_cross_below = (trigger > feature) & (trigger_d < feature_d)

    # When the previous day closing was higher than SMA but today's closing
    # is lower than SMA, cross above signal, i.e. the SMA cross
    # the price trigger (close) from being below to being above
    idx_cross_above = (trigger < feature) & (trigger_d > feature_d)

    # Signals for each crossing
    position = pd.Series(index=df.index, dtype=np.float32, data=np.nan)
    position.loc[idx_cross_below] = -1.0 # Sell if price cross SMA from below
    position.loc[idx_cross_above] = +1.0 # Buy if price cross SMA from above

    return position

def run_sma_strategy(prices: pd.DataFrame,
                     params: SMA_Params) -> pd.DataFrame:
    """Run a SMA strategy.

    Generate trading signal from an SMA strategy, considering all cost_description_
    related to execution and stock shorting

    Args:
        prices (pd.DataFrame): dataframe with closing prices of the underlying
        for each trading day
        params (SMA_Params): parameters for the SMA strategy

    Returns:
        pd.DataFrame: DataFrame with the trading strategy, including
        cost and PnL
    """
    # Compute underlying instrinsic stats: returns and volatility
    price_df=prices.assign(close_ret=lambda df: estimators.get_returns(df.close))\
                   .assign(benchmark_vol=lambda df: estimators.ewma_vol(df.close_ret))
    
    # Compute signals and features: Run short/long SMA and generate trading signals
    price_df=price_df.assign(sma_short=lambda df: estimators.moving_average(df.close, days=params.short_sma_window))\
                     .assign(sma_long=lambda df: estimators.moving_average(df.close, days=params.long_sma_window))\
                     .assign(signal=lambda df: signal_sma(df, params.feature_col, trigger_col='close'))
    
    #Apply vol scaling, scale bets according to underlying volatility
    if params.vol_scaling_func is not None:
        price_df=price_df.assign(bet_size=lambda df: df.benchmark_vol.apply( params.vol_scaling_func )*params.bet_size)
    else:
        price_df=price_df.assign(bet_size=params.bet_size) 
        
    # Compute shares held at each day and executed for each day
    # Number of shares is the betsize at each moment divided by share price
    # The amount of shares is held (ffill) while no other signal is generated
    # Observe than signal series has nan most of the time, when no buy/signal is generated
    price_df=price_df.assign(shares_held=lambda df: (df.signal*df.bet_size/df.close).ffill().fillna(value=0.0))\
                     .assign(executed_shares=lambda df: df.shares_held.diff())
    
    # Compute cost: execution, holding cost
    # The cost is the broker fee plus slippage, and eventually holding cost if we are shorting 
    # the underlying
    # The cash transaction at each time considers execution of shares plus costs
    price_df=price_df.assign(execution_cost=lambda df: (df.executed_shares.abs()*df.close*(params.execution_fee+params.mean_slippage)).fillna(value=0.0))\
                     .assign(holding_cost=lambda df: np.where(df.shares_held < 0, -1.0*df.shares_held*df.close*params.borrowing_cost*(df.index.diff().days.fillna(0))/365, 0))\
                     .assign(cash_transaction=lambda df: -df.executed_shares.fillna(value=0.0)*df.close-df.execution_cost-df.holding_cost)
    
    # Set initial investment on the , this is a funded strategy
    price_df.loc[price_df.index[0], 'cash_transaction'] = params.bet_size

    # Calculate carry and investment pnl and stategy vol
    # invested amount is the cash amount we have invested in shares (long or short)
    # the total PnL of the strategy is what we have in the cash account plus the investment account
    # Calculate return of the Strategy and its volatility
    price_df=price_df.assign(cash_amount=lambda df: compound_cash(df.cash_transaction, df.sofr))\
                     .assign(invested_amount=lambda df: df.shares_held*df.close)\
                     .assign(strat_pnl=lambda df: df.cash_amount+df.invested_amount)\
                     .assign(strat_ret=lambda df: df.strat_pnl.pct_change())\
                     .assign(strategy_vol=lambda df: estimators.ewma_vol(df.strat_ret))                       

    return price_df