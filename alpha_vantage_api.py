"""Implement basic functionality of the Alpha Vantage API.

For further reference of Alpha Vantage visit: https://www.alphavantage.co/#about

"""

# API key for victor.montielargaiz@gmail.com
from enum import Enum
from io import BytesIO
import pandas as pd
import requests

API_KEY: str = r'ELCG8TQ7BK4JS4WU'
# API_KEY : str = r'demo'

# AlphaVantage API entry point
HOST_NAME = r"https://www.alphavantage.co"


class OuputSize(Enum):
    COMPACT = 1
    FULL = 2


def get_symbol_matches(query_str: str) -> dict:
    """Returns a list of symbols given a query string.

    Parameters
    ----------

    query_str : str
        Query string of the symbol to search for

    Return
    ------
    List of best matches for the query string
    """
    url: str = f"{HOST_NAME}/query?function=SYMBOL_SEARCH&keywords={query_str}&apikey={API_KEY}"
    r = requests.get(url)
    data = r.json()

    if 'bestMatches' in data:
        return data['bestMatches']
    else:
        raise Exception(data)


def get_unadjusted_daily_prc_ts(symbol: str,
                                out_size: OuputSize = OuputSize.COMPACT) -> pd.DataFrame:
    """Returns the unadjusted daily prices for a given symbol.

    Returns a dataframe with the full available history of the prices OHLC for a given symbol

    Args:
        symbol (str): Symbol name
        out_size (OutputSize): Either COMPACT or FULL historic

    Returns:
        pd.DataFrame: Series with OHLC prices
    """
    # Get API-REST response
    output_flag: str = out_size.name.lower()
    url: str = f"{HOST_NAME}/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize={output_flag}&datatype=csv&apikey={API_KEY}"
    r = requests.get(url)

    # Read CSV, set timestamp as index and reverse the dataframe
    if r.status_code == 200:
        df = pd.read_csv(BytesIO(r.content), parse_dates=['timestamp'])\
               .set_index( 'timestamp' )\
               .iloc[::-1]
        return df
    else:
        raise Exception(r.content)
