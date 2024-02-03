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
    """_summary_
    """
    execution_fee: float
    mean_slippage: float
    borrowing_cost: float
    bet_size: float  # in cash term
    short_sma_window: int
    long_sma_window: int
    feature_col: str
    vol_scaling_func: Optional[ Callable[[float], float ]] = None #Vol Scaling function, takes vol estimation as input, returns bet size scaling within [0,1]    

def compound_cash(cashflows: pd.Series, daily_rate: pd.Series) -> pd.Series:
    """_summary_

    Args:
        cashflows (pd.Series): _description_
        daily_rate (pd.Series): _description_

    Returns:
        pd.Series: _description_
    """

    df = pd.concat(objs=[cashflows, daily_rate], axis=1)
    df.columns = ['cashflows', 'd_rate']
    df = df.assign(days=lambda df: df.index.diff().days.fillna(0))\
        .assign(interest=lambda x: x.d_rate*x.days/365)\
        .assign(compounded=0.0)

    # Set first element
    df.loc[df.index[0], 'compounded'] = df.loc[df.index[0], 'cashflows']

    # Compund each day the cash position
    for n, n_1 in zip(df.index[1:], df.index[0:-1]):
        df.loc[n, 'compounded'] = (
            df.loc[n_1, 'compounded'])*(1+df.loc[n_1, 'interest'])+df.loc[n, 'cashflows']

    return pd.Series(df.compounded)

def signal_sma(df: pd.DataFrame, feature_col: str, trigger_col: str = 'close') -> pd.Series:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        feature_col (str): _description_
        trigger_col (str, optional): _description_. Defaults to 'close'.

    Returns:
        pd.Series: _description_
    """

    feature, trigger = df[feature_col], df[trigger_col]
    feature_d, trigger_d = feature.shift(-1), trigger.shift(-1)
    idx_cross_below = (trigger > feature) & (trigger_d < feature_d)
    idx_cross_above = (trigger < feature) & (trigger_d > feature_d)
    position = pd.Series(index=df.index, dtype=np.float32, data=np.nan)
    position.loc[idx_cross_below] = -1.0
    position.loc[idx_cross_above] = +1.0

    return position

def run_sma_strategy(prices: pd.DataFrame,
                     params: SMA_Params) -> pd.DataFrame:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_
        sma_short_period (int): _description_
        sma_long_period (int): _description_
        bet_size (float): _description_
        execution_fee (float): _description_
        borrowing_cost (float): _description_
        mean_slippage (float): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # Compute underlying instrinsic stats
    price_df=prices.assign(close_ret=lambda df: estimators.get_returns(df.close))\
                   .assign(benchmark_vol=lambda df: estimators.ewma_vol(df.close_ret))
    
    # Compute signals and features
    price_df=price_df.assign(sma_short=lambda df: estimators.moving_average(df.close, days=params.short_sma_window))\
                     .assign(sma_long=lambda df: estimators.moving_average(df.close, days=params.long_sma_window))\
                     .assign(signal=lambda df: signal_sma(df, params.feature_col, trigger_col='close'))
    
    #Apply vol scaling
    if params.vol_scaling_func is not None:
        price_df=price_df.assign(bet_size=lambda df: df.benchmark_vol.apply( params.vol_scaling_func )*params.bet_size)
    else:
        price_df=price_df.assign(bet_size=params.bet_size) 
        
    # Compute shares held and executed
    price_df=price_df.assign(shares_held=lambda df: (df.signal*df.bet_size/df.close).ffill().fillna(value=0.0))\
                     .assign(executed_shares=lambda df: df.shares_held.diff())
    
    # Compute cost: execution, holding cost
    price_df=price_df.assign(execution_cost=lambda df: (df.executed_shares.abs()*df.close*(params.execution_fee+params.mean_slippage)).fillna(value=0.0))\
                     .assign(holding_cost=lambda df: np.where(df.shares_held < 0, -1.0*df.shares_held*df.close*params.borrowing_cost*(df.index.diff().days.fillna(0))/365, 0))\
                     .assign(cash_transaction=lambda df: -df.executed_shares.fillna(value=0.0)*df.close-df.execution_cost-df.holding_cost)
    
    # Set initial investment on the account
    price_df.loc[price_df.index[0], 'cash_transaction'] = params.bet_size

    # Calculate carry and investment pnl and stategy vol
    price_df=price_df.assign(cash_amount=lambda df: compound_cash(df.cash_transaction, df.sofr))\
                     .assign(invested_amount=lambda df: df.shares_held*df.close)\
                     .assign(strat_pnl=lambda df: df.cash_amount+df.invested_amount)\
                     .assign(strat_ret=lambda df: df.strat_pnl.pct_change())\
                     .assign(strategy_vol=lambda df: estimators.ewma_vol(df.strat_ret))                       

    return price_df