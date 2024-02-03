""" Estimator functions used accross the project.
"""

from enum import Enum
from typing import Optional

import pandas as pd
import numpy as np


class ReturnType(Enum):
    SIMPLE = 1
    LOG = 2


def get_returns(prc_series: pd.Series,
                ret_type: Optional[ReturnType] = ReturnType.LOG) -> pd.Series:
    """Calculate return series from a price series.

    Args:
        series (pd.Series): price series of an instrument
        ret_type (ReturnType, optional): Either simple or Log returns. 
        Defaults to ReturnType.LOG.

    Returns:
        pd.Series: Series of returns, with some NaN values at first samples
    """
    if ret_type == ReturnType.LOG:
        return np.log(prc_series/prc_series.shift(1))
    elif ret_type == ReturnType.SIMPLE:
        return prc_series.diff()/prc_series.shift(1)
    else:
        raise Exception(f"ReturnType {ret_type} not recognized")

def moving_average(prc_serie: pd.Series,
                   days: int ) -> pd.Series:
    """Calculate a moving average on a price series.

    Args:
        prc_serie (pd.Series): Series object with prices
        days (int): days of the moving average

    Returns:
        pd.Series: MA series
    """
    return prc_serie.rolling(window=days).mean()

def ewma(serie: pd.Series,
         alpha: float) -> pd.Series:
    """Calculate Exponentially Weighted moving average.

    Assumes index is a datetime with observations timestamp

    Args:
        prc_serie (pd.Series): price or return series
        alpha (float): decay parameter

    Returns:
        pd.Series: EWMA
    """
    return serie.ewm(alpha=alpha).mean()

def ewma_vol(ret_series: pd.Series,
             alpha: float = 0.03,
             trading_days: int = 255) -> pd.Series:
    """Calculate Exponentially Weighted moving vol of a return series
    
    Assumes daily observations
    Assumes index is a datetime with observations timestamp

    Args:
        ret_series (pd.Series): return series
        alpha (float): decay parameter
        trading_days (int, optional): num of trading days in a year.
        Defaults to 255.

    Returns:
        pd.Series: EWMA volatility
    """
    return ret_series.ewm(alpha=alpha).std() * np.sqrt(trading_days)