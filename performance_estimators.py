import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.stats import norm

translation = {
        'return': 'Return', 
        'vol': 'Annual Vol',
        'skewness': 'Skewness',
        'kurtosis': 'Kurtosis',
        'VaR': 'VaR',
        'sharpe_ratio': 'Sharpe',
        'max_dd': 'Max DrawDown',
        'max_dd_length' : 'Max DD Length',
        'calmar_ratio': 'Calmar',
        'stability_ts' : 'Stability Time Series',
        'omega_ratio': 'Omega',
        'sortino_ratio' : 'Sortino',
        'tail_ratio': 'Tail Ratio',
        'common_sense_ratio': 'Common Sense Ratio',
        'peak_to_trough_maxdd': "Peak to Trough Max DrawDown",
        'peak_to_peak_maxdd': "Peak To Peak Max Drawdown",
        'peak_to_peak_longest': "Peak to peak Longest",
        'var_pctl': 'VaR Percentile',
        'risk_free': 'Risk Free Rate',
        'trades_per_year': 'Avg Trades per Year',
        'target_return': 'Target Return'
    }

def _convert_to_array(x):
    v = np.asanyarray(x)
    v = v[np.isfinite(v)]
    return v

TRADING_DAY_MSG = 'Trading days needs to be > 0'

def annual_return(returns, price_rets='price', trading_days=252):
    '''
    Computing the average compounded return (yearly)
    '''
    assert trading_days > 0, TRADING_DAY_MSG
    v = _convert_to_array(returns)
    n_years = v.size / float(trading_days)
    if price_rets == 'strategy_returns':
        return (np.prod((1. + v) ** (1. / n_years)) - 1. if v.size > 0 else np.nan)
    else:
        return (np.sum(v) * (1. / n_years) if v.size > 0 else np.nan)

def annual_volatility(returns, trading_days=252):
    assert trading_days > 0, TRADING_DAY_MSG
    v = _convert_to_array(returns)
    return (np.std(v) * np.sqrt(trading_days) if v.size > 0 else np.nan)


def value_at_risk(returns, horizon=10, pctile=0.99, mean_adj=False):
    assert horizon > 1, 'horizon>1'
    assert pctile < 1
    assert pctile > 0, 'pctile in [0,1]'
    v = _convert_to_array(returns)
    stdev_mult = norm.ppf(pctile)  # i.e., 1.6449 for 0.95, 2.326 for 0.99
    if mean_adj:
        gains = annual_return(returns, 'price', horizon)
    else:
        gains = 0

    return (np.std(v) * np.sqrt(horizon) * stdev_mult - gains if v.size > 0 else np.nan)


def sharpe_ratio(returns, risk_free=0., trading_days=252):
    assert trading_days > 0, TRADING_DAY_MSG
    v = _convert_to_array(returns - risk_free)
    return (np.mean(v) / np.std(v) * np.sqrt(trading_days) if v.size > 0 else np.nan)


def max_drawdown(returns, price_rets='price'):
    v = _convert_to_array(returns)
    if v.size == 0:
        return np.nan
    if price_rets == 'strategy_returns':
        cumret = np.concatenate(([100.], 100. * np.cumprod(1. + v)))
        maxret = np.fmax.accumulate(cumret)
        return np.nanmin((cumret - maxret) / maxret)
    else:
        cumret = np.concatenate(([1.], np.cumsum(v)))
        maxret = np.fmax.accumulate(cumret)
        return np.nanmin(cumret - maxret)


def max_drawdown_length(returns, price_rets='price'):
    v = _convert_to_array(returns)
    if v.size == 0:
        return np.nan
    drawndown = np.zeros(len(v) + 1)
    dd_dict = dict()
    if price_rets == 'strategy_returns':
        cumret = np.concatenate(([100.], 100. * np.cumprod(1. + v)))
        maxret = np.fmax.accumulate(cumret)
        drawndown[(cumret - maxret) / maxret < 0] = 1
    else:
        cumret = np.concatenate(([1.], np.cumsum(v)))  # start at one? no matter
        maxret = np.fmax.accumulate(cumret)
        drawndown[(cumret - maxret) < 0] = 1

    f = np.frompyfunc((lambda x, y: (x + y) * y), 2, 1)
    run_lengths = f.accumulate(drawndown, dtype='object').astype(int)

    trough_position = np.argmin(cumret - maxret)
    peak_to_trough = run_lengths[trough_position]

    next_peak_rel_position = np.argmin(run_lengths[trough_position:])
    next_peak_position = next_peak_rel_position + trough_position

    if run_lengths[next_peak_position] > 0:  # We are probably still in DD
        peak_to_peak = np.nan
    else:
        peak_to_peak = run_lengths[next_peak_position - 1]
        # run_lengths just before it hits 0 (back to peak) is the
        # total run_length of that DD period.

    longest_dd_length = max(run_lengths)  # longest, not nec deepest

    dd_dict['peak_to_trough_maxdd'] = peak_to_trough
    dd_dict['peak_to_peak_maxdd'] = peak_to_peak
    dd_dict['peak_to_peak_longest'] = longest_dd_length
    return dd_dict


def calmar_ratio(returns, price_rets='price', trading_days=252):
    assert trading_days > 0, TRADING_DAY_MSG
    v = _convert_to_array(returns)
    if v.size == 0:
        return np.nan
    maxdd = max_drawdown(v, price_rets)
    if np.isnan(maxdd):
        return np.nan
    annret = annual_return(v, price_rets, trading_days=trading_days)
    return annret / np.abs(maxdd)


def stability_of_timeseries(returns, price_rets='price'):
    v = _convert_to_array(returns)
    if v.size == 0:
        return np.nan
    if price_rets == 'strategy_returns':
        v = np.cumsum(np.log1p(v))
    else:
        v = np.cumsum(v)
    lin_reg = linregress(np.arange(v.size), v)
    return lin_reg.rvalue ** 2


def omega_ratio(returns, risk_free=0., target_return=0., trading_days=252):
    assert trading_days > 0, TRADING_DAY_MSG
    v = _convert_to_array(returns)
    if v.size == 0:
        return np.nan
    return_thresh = (1. + target_return) ** (1. / trading_days) - 1.
    v = v - risk_free - return_thresh
    numer = np.sum(v[v > 0.])
    denom = -np.sum(v[v < 0.])
    return (numer / denom if denom > 0. else np.nan)


def sortino_ratio(returns, target_return=0., trading_days=252):
    assert trading_days > 0, TRADING_DAY_MSG
    v = _convert_to_array(returns)
    if v.size == 0:
        return np.nan
    v = v - target_return
    downside_risk = np.sqrt(np.mean(np.square(np.clip(v, np.NINF, 0.))))
    return np.mean(v) * np.sqrt(trading_days) / downside_risk


def tail_ratio(returns):
    v = _convert_to_array(returns)
    if v.size > 0:
        try:
            return np.abs(np.percentile(v, 95.)) / np.abs(np.percentile(v, 5.))
        except FloatingPointError:
            return np.nan
    else:
        return np.nan

def common_sense_ratio(returns):
    # This cannot be compared with pyfolio routines because they implemented a
    # wrong formula CSR = Tail Ratio * Gain-to-Pain Ratio
    # and Gain-to-Pain Raio = Sum(Positive R) / |Sum(Negative R)|
    v = _convert_to_array(returns)
    if v.size == 0:
        return np.nan
    return tail_ratio(returns) * np.sum(v[v > 0.]) / np.abs(np.sum(v[v < 0.]))

def avg_trades_per_year(trades: pd.Series) -> float:
    """Return average number of trades per year

    Args:
        trades (pd.Series): _description_

    Returns:
        float: _description_
    """
    num_trades= trades.sum()
    years = (trades.index[-1] - trades.index[0]).days / 365.0
    return num_trades / years

def calc_performance(df: pd.DataFrame,
                     return_field: str,
                     params: dict,
                     price_rets: str = 'price') -> dict:
    """Run performance calculation on a strategy dataframe

    Args:
        df (pd.DataFrame): 
            dataframe with strategy data, with standarized 
            columns name for this project
        return_field (str):
            name of the field with return data
        price_rets (str, optional): _description_. Defaults to 'price'.
            Either 'price' or 'strategy_returns' to measure
            performance of investment in instrument or in financed
            strategy
        params (dict):
            Parameters for various performance metrics

    Returns:
        dict: dictionary with key performance measures
    """
    assert price_rets in ['price', 'strategy_returns'], f"Invalid price_rets: {price_rets}"

    var_pctl = params[ 'var_pctl' ]
    risk_free = params[ 'risk_free' ]
    target_return = params[ 'target_return' ]

    perf = { }
    perf['return'] = annual_return(df[return_field], price_rets=price_rets)
    perf['vol'] = annual_volatility(df[return_field])
    perf['skewness'] = df[return_field].skew()
    perf['kurtosis'] = df[return_field].kurtosis()
    perf['VaR'] = value_at_risk(df[return_field], horizon=10, pctile=var_pctl)
    perf['sharpe_ratio'] = sharpe_ratio(df[return_field], risk_free=risk_free)
    perf['max_dd'] = max_drawdown(df[return_field], price_rets=price_rets)
    
    # Statistics for DD length
    for k, v in max_drawdown_length(df[return_field], price_rets=price_rets).items():
        perf[k]=v

    perf['calmar_ratio'] = calmar_ratio(df[return_field], price_rets=price_rets)
    perf['stability_ts'] = stability_of_timeseries(df[return_field], price_rets=price_rets)
    perf['omega_ratio'] = omega_ratio(df[return_field], risk_free=risk_free, target_return=target_return)
    perf['sortino_ratio'] = sortino_ratio(df[return_field], target_return=target_return)
    perf['tail_ratio'] = tail_ratio(df[return_field])
    perf['common_sense_ratio'] = common_sense_ratio(df[return_field])
    
    if price_rets == 'strategy_returns':
        perf['trades_per_year'] = avg_trades_per_year(~df.signal.isnull())

    perf['params'] = params

    return perf

def performance_df( perf_stats: dict, name:str) -> pd.DataFrame:
    """Return performance metrix as dataframe

    Args:
        perf_stats (dict): dict with performance statistics,
        as returned by calc_performance
        name (dict): name of the series

    Returns:
        pd.DataFrame: Performance as a dataframe
    """
    global translation
    
    #Perform tranlation of metrics
    out = { translation[k] : v for k,v in perf_stats.items() if k not in ['params']}
    df = pd.DataFrame.from_dict( out, orient='index' )
    df.columns = [name]
    
    return df
    
def pretty_print_performance( perf_stats: dict ) -> None:
    """Pretty print a performance statistic results

    Takes the return of calc_performance calculation and
    print formatter results

    Args:
        perf_stats (dict): dict with performance statistics,
        as returned by calc_performance
    """
    global translation

    for k,v in perf_stats.items():

        match k :  
            case "return"|"vol"|'max_dd'|"VaR": 
                print(f"{translation[k]}: {v:0.2%}")
            case "sharpe_ratio"|"calmar_ratio"|"omega_ratio"\
                |"sortino_ratio"|"tail_ratio"|"common_sense_ratio"\
                |"stability_ts":
                print(f"{translation[k]}: {v:0.2f}")
            case "max_dd_length":
                print(f"{translation[k]}:")
                for k_m, v_m in v.items():
                    print(f"**{translation[k_m]}: {v_m}")
            case "peak_to_trough_maxdd"|"peak_to_peak_maxdd"\
                |"peak_to_peak_longest":
                print(f"{translation[k]}: {v}")
            case 'skewness' | 'kurtosis' | 'trades_per_year':
                print(f"{translation[k]}: {v:0.2f}")
            case 'params':
                print(f"Statistic Params:")
                for k_m, v_m in v.items():
                    print(f"**{translation[k_m]}: {v_m:0.2%}")
            case _ : 
                print(f"Unexpected metrix {k}: {v:0.2f}")
