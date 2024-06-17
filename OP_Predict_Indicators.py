import pandas as pd
import numpy as np

from OP_Predict_Config import *

# Append the target row to the dataframe
# We want to predict 1 period into the future
# So we add one period into the future to build 
# the dataset during the data preparation phase.
def appendTargetRow(df):
    last_row = df.iloc[[-1]]

    a = df.index[-1].to_pydatetime()    
    b = a + dt.timedelta(0, 300) # days, seconds, then other fields.

    last_row = pd.DataFrame(last_row)
    last_row['Date'] = b
    last_row.set_index('Date', inplace=True) 
        
    return df

# Add common buy/sell triggers to the dataframe
# to pre-weight the model
def addTickerTriggers(df):
    # Set some signals from the RSI
    signal = df['adj_sig_13']
    buy = np.where(signal < .3, 0.1, 0)
    sell = np.where(signal > .7, 0.1, 0)
    df["rsi_sell"] = sell
    df["rsi_buy"] = buy

    return df


def get_row_features(df):    
    # Remove all of the feature and index labels for training
    features = df.iloc[:,LOOKBACK_COLS:]
    features = features.reset_index()
    cols = [0]
    features.drop(features.columns[cols], axis=1, inplace=True)
    column_stop =  features.shape[1] 
    features.columns = range(column_stop)
    return features



def get_df_features(df, train_split, predict=False):    
    
    # Remove all of the feature and index labels for training
    features = df.iloc[:,LOOKBACK_COLS:]
    features = features.reset_index()
    cols = [0]
    features.drop(features.columns[cols], axis=1, inplace=True)
    
    if predict == False:
        train_data = features.loc[0 : train_split - 1]
        test_data = features.iloc[train_split:]
    else:
        predict_data = features.tail(PAST)
        features = features[:-PAST]

        train_data = features.loc[0 : train_split - 1]
        test_data = features.iloc[train_split:]     

    column_stop =  features.shape[1] 
    features.columns = range(column_stop)
    train_data.columns = range(column_stop)
    test_data.columns = range(column_stop)
    

    if predict == False:
        #TODO: Create a separate split of the train data
        return train_data, test_data, features
    else:
        predict_data.columns = range(column_stop)
        return train_data, test_data, features, predict_data

# Extract the target column and set it at
# the start of the series of columns that will be used for the analysis
def appendTickerHist(df: pd.DataFrame):
    
    # Create the target data
    # set the flag to 1 if the current data is higher than the previous  ticker close
    priceDiff = df['adj_mkt_cap'].diff()
    priceDiff = np.where(priceDiff < 0, 0, 1)
    df["direction"] = priceDiff

    # We want this column to be first
    first_column = df.pop('direction') 
    df.insert(PREV_HIST+1, 'direction', first_column) 

    df.dropna(inplace=True)
    # df.tail()

    first_column = df.pop('adj_mkt_cap') # Remove the market cap target
    return df


def interpolateTickerdataRevsersed(df: pd.DataFrame, resolution: str = "day", upsample: str ="1min"):
    
    resolution = "day"
    df[resolution] = df.index.day_of_year

    interpolated_days = []
    periods = df[resolution].unique()
    # print(len(periods))
    for period in periods:
        df_cap_day = df.loc[df[resolution] == period]     
        if  (df_cap_day.shape[0] > 3)   :
            df_cap_day = df_cap_day.reindex(pd.date_range(df_cap_day.index[0], df_cap_day.index[-1], freq=upsample))
            # print(df_cap_day.shape)
            df_cap_day = df_cap_day.interpolate(method='linear')
            interpolated_days.append(df_cap_day)

    df = pd.concat(interpolated_days)
    print(df.shape)
    return df

# Upsample the ticker data from the default interpolation
# Use the polynomial of bspline for better results
def interpolateTickerdata(df: pd.DataFrame, resolution: str = "day", upsample: str ="1min"):
    
    df["day"] = df.index.day_of_year

    interpolated_days = []
    periods = df[resolution].unique()

    for period in periods:
        df_cap_day = df.loc[df[resolution] == period]        
        df_cap_day = df_cap_day.reindex(pd.date_range(df_cap_day.index[0], df_cap_day.index[-1], freq=upsample))
        if (upsample == "60min"):
            df_cap_day = df_cap_day.interpolate(method='linear')
        else:
            # df_cap_day = df_cap_day.interpolate(method='polynomial', order=3)
            df_cap_day = df_cap_day.interpolate(method='linear')
        interpolated_days.append(df_cap_day)

    df = pd.concat(interpolated_days)

    # Drop the column that was appended earlier
    cols = [-1]
    df.drop(df.columns[cols],axis=1,inplace=True)

    print(df.shape)
    return df


def simpleAnalysis(df, csv=False, window_1 = 4, window_2 = 8, window_3 = 16,  window_4 = 32):    

    if csv == True:        
        df["close_price"] = pd.to_numeric(df["Close/Last"].map(lambda x: x.lstrip("$")))
        df["high_price"] = pd.to_numeric(df["High"].map(lambda x: x.lstrip("$")))
        df["low_price"] = pd.to_numeric(df["Low"].map(lambda x: x.lstrip("$")))
        df["open_price"] = pd.to_numeric(df["Open"].map(lambda x: x.lstrip("$")))
        df["volume"] = pd.to_numeric(df["Volume"])

    else:
        df["close_price"] = pd.to_numeric(df["close_price"])
        df["high_price"] = pd.to_numeric(df["high_price"])
        df["low_price"] = pd.to_numeric(df["low_price"])
        df["open_price"] = pd.to_numeric(df["open_price"])
        df["volume"] = pd.to_numeric(df["volume"])
    # df.loc[:, 'close_price'].mean()
    
    # Calculate 14-period RSI
    df['rsi_s']  = rsi(df, window_2)
    df['rsi_m']  = rsi(df, window_3)
    df['rsi_l']  = rsi(df, window_4)

    # Calculate 10-minute Exponential Moving Average EMA)
    df['ema'] = df['close_price'].ewm(com=4, adjust=False).mean()

    # Calculate 30-day Simple Moving Average (SMA)
    df['sma'] = df['close_price'].rolling(8).mean()

    if csv == True:      
        df.set_index('Date', inplace=True)    
        cols = [0,1,2,3,4]
        df.drop(df.columns[cols],axis=1,inplace=True)
    else:
        cols = [5,6,7]
        df.drop(df.columns[cols],axis=1,inplace=True)
        # df.set_index('begins_at', inplace=True)    

    # Calculating the short-window (7 days) simple moving average
    # Calculating the long-window (15 days) simple moving average

    df['sma_s'] = df['close_price'].rolling(window_1).mean()
    df['sma_l'] = df['close_price'].rolling(window_2).mean()

    df['ema_s'] = df['close_price'].ewm( com = window_1, adjust = False ).mean()
    df['ema_l'] = df['close_price'].ewm( com = window_2, adjust = False ).mean()

    # Do the high price
    df['ema_s_high'] = df['high_price'].ewm( com = window_2, adjust = False ).mean()
    df['ema_l_high'] = df['high_price'].ewm( com = window_3, adjust = False ).mean()

    # Do the high price
    df['ema_s_low'] = df['low_price'].ewm( com = window_3, adjust = False ).mean()
    df['ema_l_low'] = df['low_price'].ewm( com = window_4, adjust = False ).mean()

    df['ema_s_open'] = df['open_price'].ewm( com = window_3, adjust = False ).mean()
    df['ema_l_open'] = df['open_price'].ewm( com = window_4, adjust = False ).mean()

    df["volume_rolling"] = df['volume'].rolling(window_1).mean()

    return df




def appendTickerHistory(df: pd.DataFrame, columnName: str, history: int = 1, average: float = 0):
    """
    Add the history column-wise to the data
    The ML model can have better access to history data outside of the indications
    Call it in a loop to hitory vales, i.e. it will append history columns to the 
    incoming dataset
    """
    def demeanSeriesInline(df: pd.DataFrame, columnName, shift: int = 1, average: float = 0):
        
        #referemce
        if (average == 0):
            average = df[columnName].mean()

        df_cap_average = df[columnName] / average   
        market_cap = df_cap_average[:-1]    

        adj_series = df[columnName] / average    
        first_valid_adj_series = (adj_series / adj_series.loc[ adj_series.first_valid_index() ])

        demeaned_adj_series = first_valid_adj_series.pct_change(periods=1).shift(-shift)
        demeaned_adj_signal = demeaned_adj_series.copy()
        demeaned_adj_signal = demeaned_adj_signal.dropna()
        demeaned_adj_signal = demeaned_adj_signal[demeaned_adj_signal.index.isin( market_cap.index )]
        df[len(df.columns)]  = demeaned_adj_signal

        return demeaned_adj_signal

    for i in range(history):
        demeanSeriesInline(df, columnName, i, average)

    return df


def rsi(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    """See source https://github.com/peerchemist/finta
    and fix https://www.tradingview.com/wiki/Talk:Relative_Strength_Index_(RSI)
    Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements.
    RSI oscillates between zero and 100. Traditionally, and according to Wilder, RSI is considered overbought when above 70 and oversold when below 30.
    Signals can also be generated by looking for divergences, failure swings and centerline crossovers.
    RSI can also be used to identify the general trend."""

    delta = ohlc["close_price"].diff()

    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    _gain = up.ewm(com=(period - 1), min_periods=period).mean()
    _loss = down.abs().ewm(com=(period - 1), min_periods=period).mean()

    RS = _gain / _loss
    return pd.Series(100 - (100 / (1 + RS)), name="RSI")



def check_rsi(ohlc: pd.DataFrame):
    print(ohlc)
    low_zero = (ohlc < 0).any()
    high_cem = (ohlc < 0).any()

    if (low_zero or high_cem):
        print("an error occured during the calculation")
        return False
    
    return True



def generateModelIndicators(df: pd.DataFrame, period: int = 1):
    # Manipulate raw data
    # NOTE: .sum() has axis 1 because we want the sum of the column which is just
    #       1 ticker symbol.
    #       .mean() will then take the mean average of all the ticker symbol 
    average_market_cap = df["close_price"].mean()

    average_market_vol = df["volume"].mean()

    # Average
    short_swma_average = df['sma_s'] / average_market_cap
    long_swma_average = df['sma_l'] / average_market_cap
    short_ewma_average = df['ema_s'] / average_market_cap
    long_ewma_average = df['ema_l'] / average_market_cap
    
    # Add for other candle data
    long_ewma_lS_average = df['ema_s_low'] / average_market_cap
    long_ewma_hS_average = df['ema_s_high'] / average_market_cap
    long_ewma_oS_average = df['ema_s_open'] / average_market_cap
    
    # Add for other candle data
    long_ewma_lL_average = df['ema_l_low'] / average_market_cap
    long_ewma_hL_average = df['ema_l_high'] / average_market_cap
    long_ewma_oL_average = df['ema_l_open'] / average_market_cap

    rsi_s_average = df['rsi_s'] / 100
    rsi_m_average = df['rsi_m'] / 100
    rsi_l_average = df['rsi_l'] / 100
    df_cap_average = df["close_price"] / average_market_cap    
    df_vol_average = df["volume_rolling"] / average_market_vol
        
    # By looking at the debug cells below, we see that both heads for the long and short de-meaned pandas Dataframes have NaN 
    # (Not-a-number.)       
    # Going to only start with the first valid number Using .first_valid_index
    # https://stackoverflow.com/questions/42137529/pandas-find-first-non-null-value-in-column
    short_swma_average_first_valid = (short_swma_average  / short_swma_average.loc[ short_swma_average.first_valid_index() ])
    long_swma_average_first_valid = (long_swma_average / long_swma_average.loc[ long_swma_average.first_valid_index() ])
    short_ewma_average_first_valid = (short_ewma_average  / short_ewma_average.loc[ short_ewma_average.first_valid_index() ])
    long_ewma_average_first_valid = (long_ewma_average / long_ewma_average.loc[ long_ewma_average.first_valid_index() ])

    # Add for other candle data
    long_ewma_lS_average_first_valid = (long_ewma_lS_average / long_ewma_lS_average.loc[ long_ewma_lS_average.first_valid_index() ])
    long_ewma_hS_average_first_valid = (long_ewma_hS_average / long_ewma_hS_average.loc[ long_ewma_hS_average.first_valid_index() ])
    long_ewma_oS_average_first_valid = (long_ewma_oS_average / long_ewma_oS_average.loc[ long_ewma_oS_average.first_valid_index() ])

    long_ewma_lL_average_first_valid = (long_ewma_lL_average / long_ewma_lL_average.loc[ long_ewma_lL_average.first_valid_index() ])
    long_ewma_hL_average_first_valid = (long_ewma_hL_average / long_ewma_hL_average.loc[ long_ewma_hL_average.first_valid_index() ])
    long_ewma_oL_average_first_valid = (long_ewma_oL_average / long_ewma_oL_average.loc[ long_ewma_oL_average.first_valid_index() ])

    # volume
    long_vol_average_first_valid = (df_vol_average / df_vol_average.loc[ df_vol_average.first_valid_index() ])

    rsi_s_average_first_valid = (rsi_s_average / rsi_s_average.loc[ rsi_s_average.first_valid_index() ])
    rsi_m_average_first_valid = (rsi_m_average / rsi_m_average.loc[ rsi_m_average.first_valid_index() ])
    rsi_l_average_first_valid = (rsi_l_average / rsi_l_average.loc[ rsi_l_average.first_valid_index() ])

    # De-mean
    # https://www.youtube.com/watch?v=E5PZR4YpBtM
    short_swma_demeaned = short_swma_average_first_valid.pct_change(periods=1).shift(period)
    long_swma_demeaned = long_swma_average_first_valid.pct_change(periods=1).shift(period)
    short_ewma_demeaned = short_ewma_average_first_valid.pct_change(periods=1).shift(period)
    long_ewma_demeaned = long_ewma_average_first_valid.pct_change(periods=1).shift(period)

    # Add for other candle data
    long_ewma_lS_demeaned = long_ewma_lS_average_first_valid.pct_change(periods=1).shift(period)
    long_ewma_hS_demeaned = long_ewma_hS_average_first_valid.pct_change(periods=1).shift(period)
    long_ewma_oS_demeaned = long_ewma_oS_average_first_valid.pct_change(periods=1).shift(period)

    long_ewma_lL_demeaned = long_ewma_lL_average_first_valid.pct_change(periods=1).shift(period)
    long_ewma_hL_demeaned = long_ewma_hL_average_first_valid.pct_change(periods=1).shift(period)
    long_ewma_oL_demeaned = long_ewma_oL_average_first_valid.pct_change(periods=1).shift(period)
    
    # volume
    long_vol_oL_demeaned = long_vol_average_first_valid.pct_change(periods=1).shift(period)

    rsi_s_demeaned = rsi_s_average_first_valid.pct_change(periods=1).shift(period)  
    rsi_m_demeaned = rsi_m_average_first_valid.pct_change(periods=1).shift(period)  
    rsi_l_demeaned = rsi_l_average_first_valid.pct_change(periods=1).shift(period)  

    # Drop last row
    market_cap = df_cap_average[:-1]    

    signal_1 = short_swma_demeaned.copy()
    signal_2 = long_swma_demeaned.copy()
    signal_3 = short_ewma_demeaned.copy()
    signal_4 = long_ewma_demeaned.copy()
    
    # RSI Adjusted
    signal_5 = rsi_s_demeaned.copy()
    signal_13 = rsi_m_demeaned.copy()
    signal_14 = rsi_l_demeaned.copy()

    # Add for other candle data
    signal_6 = long_ewma_lS_demeaned.copy()
    signal_7 = long_ewma_hS_demeaned.copy()
    signal_8 = long_ewma_oS_demeaned.copy()

    signal_9 = long_ewma_lL_demeaned.copy()
    signal_10 = long_ewma_hL_demeaned.copy()
    signal_11 = long_ewma_oL_demeaned.copy()

    signal_12 = long_vol_oL_demeaned.copy()

    # Drop not-a-numbers ++++++++++++++++++++++++++++++++++
    signal_1 = signal_1.dropna()
    signal_2 = signal_2.dropna()
    signal_3 = signal_3.dropna()
    signal_4 = signal_4.dropna()
    
    # RSI Adjusted
    signal_5 = signal_5.dropna()
    signal_13 = signal_13.dropna()
    signal_14 = signal_14.dropna()

    signal_6 = signal_6.dropna()
    signal_7 = signal_7.dropna()
    signal_8 = signal_8.dropna()

    signal_9 = signal_9.dropna()
    signal_10 = signal_10.dropna()
    signal_11 = signal_11.dropna()

    signal_12 = signal_12.dropna()
    
    market_cap_raw = market_cap
    market_cap = market_cap[ market_cap.index.isin(signal_1.index) & 
                            market_cap.index.isin(signal_2.index) & 
                            market_cap.index.isin(signal_4.index) & 
                            market_cap.index.isin(signal_4.index) & 
                            market_cap.index.isin(signal_5.index) & 
                            market_cap.index.isin(signal_13.index) & 
                            market_cap.index.isin(signal_14.index)
                            ]
    signal_1 = signal_1[signal_1.index.isin( market_cap.index )]
    signal_2 = signal_2[signal_2.index.isin( market_cap.index )]
    signal_3 = signal_3[signal_3.index.isin( market_cap.index )]
    signal_4 = signal_4[signal_4.index.isin( market_cap.index )]

    # RSI Adjusted
    signal_5 = signal_5[signal_5.index.isin( market_cap.index )]
    signal_13 = signal_13[signal_13.index.isin( market_cap.index )]
    signal_14 = signal_14[signal_14.index.isin( market_cap.index )]

    signal_6 = signal_6[signal_6.index.isin( market_cap.index )]
    signal_7 = signal_7[signal_7.index.isin( market_cap.index )]
    signal_8 = signal_8[signal_8.index.isin( market_cap.index )]

    signal_9 = signal_9[signal_9.index.isin( market_cap.index )]
    signal_10 = signal_10[signal_10.index.isin( market_cap.index )]
    signal_11 = signal_11[signal_11.index.isin( market_cap.index )]

    signal_12 = signal_12[signal_12.index.isin( market_cap.index )]

    df['adj_mkt_cap'] = market_cap
    df['adj_sig_1']  = signal_1
    df['adj_sig_2']  = signal_2
    df['adj_sig_3']  = signal_3
    df['adj_sig_4']  = signal_4
    df['adj_sig_5']  = signal_5
    df['adj_sig_13']  = signal_13
    df['adj_sig_14']  = signal_14

    df['adj_sig_6']  = signal_6
    df['adj_sig_7']  = signal_7
    df['adj_sig_8']  = signal_8

    df['adj_sig_9']  = signal_9
    df['adj_sig_10']  = signal_10
    df['adj_sig_11']  = signal_11

    # df['shift_cp']  = df["close_price"].shift(period) # to check that we are predicting based on the correct value

    return df, average_market_cap 


def simpleAnalysis_lseq(df, csv=False, window_1 = 4, window_2 = 8, window_3 = 16,  window_4 = 32):    
  
    # Calculate 14-period RSI
    # df['rsi_s']  = rsi(df, window_2)
    # df['rsi_m']  = rsi(df, window_3)
    # df['rsi_l']  = rsi(df, window_4)

    # Calculate 10-minute Exponential Moving Average EMA)
    df[6] = df[0].ewm(com=4, adjust=False).mean()

    # Calculate 30-day Simple Moving Average (SMA)
    df[7] = df[0].rolling(8).mean()

    df[8] = df[1].rolling(window_1).mean()
    df[9] = df[2].rolling(window_2).mean()

    df[10] = df[3].ewm( com = window_1, adjust = False ).mean()
    df[11] = df[4].ewm( com = window_2, adjust = False ).mean()

    # Do the high price
    df[12] = df[0].ewm( com = window_2, adjust = False ).mean()
    df[13] = df[1].ewm( com = window_3, adjust = False ).mean()

    # Do the high price
    df[14] = df[2].ewm( com = window_3, adjust = False ).mean()
    df[15] = df[3].ewm( com = window_4, adjust = False ).mean()

    df[16] = df[4].ewm( com = window_3, adjust = False ).mean()
    df[17] = df[5].ewm( com = window_4, adjust = False ).mean()
    

    return df

