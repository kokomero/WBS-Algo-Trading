import pathlib
import numpy as np
import pandas as pd
import yfinance as yf

def load_sofr_dataset(file_path:pathlib.Path) -> pd.DataFrame:
    """Loads the file with the SOFR historical.

    Values, including backward historical reconstruction, have been obtained from
    SOFR Index
    https://www.newyorkfed.org/markets/reference-rates/sofr#:~:text=Each%20business%20day%2C%20the%20New,by%20the%20New%20York%20Fed.

    Historical SOFR
    https://www.newyorkfed.org/markets/opolicy/operating_policy_180309

    Args:
        file_path (pathlib.Path): File to the CSV file where historical SOFR
        is stored

    Returns:
        pd.DataFrame: DataFrame with historical SOFR values, the index
        of the dataframe is the timestamp of the value, and sofr columns
        store the SOFR rate for that date
    """
    sofr = pd.read_csv( file_path, 
                        sep=';', 
                        parse_dates=['Effective Date'], 
                        dayfirst=False,
                        dtype={ 'Rate (%)': np.float64} )\
            .rename( columns={'Effective Date' : 'timestamp',
                            'Rate (%)' : 'sofr'})\
            .assign( sofr = lambda df: df.sofr / 100)\
            .set_index( 'timestamp')\
            .drop( [ 'Rate Type' ], axis = 1)
    
    #A dd multi-index column to have same formas as prices
    sofr.columns = pd.MultiIndex.from_tuples( [('Adj Close', 'sofr')] )
        
    return sofr


def load_spot_dataset(file_path: pathlib.Path) -> pd.DataFrame:
    """Read historical prices downloaded from yahoo finance and stored in a file.

    Args:
        file_path (pathlib.Path): Path pointing to the file were
        historical data for a list of underlying is stored.

        The CSV file is a multi-index colum, containing multiples underlyigns
        and multiple fields for each underlying

    Returns:
        pd.DataFrame: DataFrame with MultiIndex columns. First level has the
        field name, while second level index contains the underlying of the data
    """
    hist_read= pd.read_csv( file_path, 
                   sep = ';',
                   parse_dates = True,
                   dayfirst=False,
                   header=[0,1,2])
    
    # Yahoo finance returns a multiindex when downloading different stocks
    # Remove multindex for first column, and set that columns as
    # timestamp index
    hist_read.columns = hist_read.columns.droplevel(2)
    hist_read = hist_read.set_index( hist_read.iloc[:,0] )
    hist_read.index.name = 'timestamp'
    hist_read = hist_read.drop( columns=hist_read.columns[0] )
    hist_read.index = pd.to_datetime(hist_read.index)
    return hist_read

def get_historical_prices( prices: pd.DataFrame,
                           underlying: str,
                           fields: list[str],
                           funding_name: str = None) -> pd.DataFrame:
    """Return dataframe with market data and optionally
    a funding rate

    Args:
        prices (df.DataFrame): 
            Raw dataframe with market data for all underlyings
        underlying (str): 
            Symbol for the underlying
        fields (list[str]): 
            List of fields to retrieve, such as Adj Close, Volume...
        funding_name (str): 
            Optional, name of the funding rate with closing prices

    Returns:
        pd.DataFrame: dataframe with selected market data fields for the
        underlying and, if specified, closing levels for the
        funding rate indes

        Drop rows for which there are no data for any of the columns
        
    """
    # Add all fields needed for the underlyings
    multi_columns = [(f, underlying) for f in fields]

    # Filter the dataframe
    df = prices.loc[:, pd.MultiIndex.from_tuples(multi_columns)]

    # Add interest rate index
    if funding_name is not None:
        funding_df = pd.DataFrame(prices.loc[:, ('Adj Close', funding_name)])
        funding_df.columns = funding_df.columns.swaplevel(0, 1)
        df = df.merge( funding_df, left_index=True, right_index=True)
    
    df.columns = df.columns.droplevel(1)
    df = df.dropna()

    return df