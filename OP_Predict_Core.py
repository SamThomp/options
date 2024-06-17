from OP_Predict_Config import *
from OP_Predict_Util import *
from OP_Predict_Indicators import *
from OP_Predict_RHConnect import *
from OP_Predict_Model import *


def Predict(model, tickers):
    first_pass = False
    single_df = pd.DataFrame([])
    
    for symbol in tickers:        
        single_ticker_df, marketCap = PrepareTickerData(ticker=symbol, target_col="close_price", queryType="long", upsample="60min", predict=True)
        single_ticker_df = appendTargetRow(single_ticker_df)
        single_ticker_df = appendTickerHist(single_ticker_df)
        single_df = pd.concat([single_df, single_ticker_df])

        # Split the training data
        train_split = int(.50 * int(single_df.shape[0]))        # SPLIT_FRACTION_PREDICT
        train_data, test_data, features, predict_data = get_df_features(single_df, train_split, predict=True)
        column_stop =  features.shape[1]
        x_train, y_train, x_test, y_test, sequence_length = split_target_data(train_data, test_data, column_stop)
        dataset_train, dataset_val = get_keras_datasets(x_train, y_train, x_test, y_test, sequence_length)
        model, history = evalModel(model, dataset_train, dataset_val, first_pass)

    return model, history, dataset_val, test_data, predict_data, column_stop


# Train the model on data for different tickers
def trainModel_Multi():
    '''
    Prepare the training data for the model.
    Omit the target stock from this group so that the mode 
    can be evaluated later on with the fresh data
    '''
    tickers = ['NDAQ', 'SPY', 'XLF', 'LCID', 'VEA', 'MSFT', 'VOO', 'AMD', 'AMZN', 'META','SMCI','CRM', 'GOOG', 'COST', 'SBUX','KLAC','AXP','ANET','GE','INTC']
    ticker_df = pd.DataFrame([])
    features = pd.DataFrame([])

    first_pass = True
    model = None
    
    if (checkDailyQuery() == True):
        for symbol in tickers:        
            ticker_df, marketCap = PrepareTickerData(ticker=symbol, target_col="close_price", queryType="long", upsample="60min", predict=False) # 1min short
            ticker_df = appendTickerHist(ticker_df)

            # Split the training data
            train_split = int(SPLIT_FRACTION_TRAIN * int(ticker_df.shape[0]))
            train_data, test_data, features = get_df_features(ticker_df, train_split, False)
            column_stop =  features.shape[1]
            x_train, y_train, x_test, y_test, sequence_length = split_target_data(train_data, test_data, column_stop)
            dataset_train, dataset_val = get_keras_datasets(x_train, y_train, x_test, y_test, sequence_length)
            model, history = evalModel(model, dataset_train, dataset_val, first_pass)
            first_pass = False
    else:
        model = keras.models.load_model("models/OP_classify.keras")

    return model, features


def PrepareTickerData(ticker: str = "SPY", target_col: str = "close_price", queryType: str = "long", upsample: str ="1min", predict: bool = False):
    '''
    Prepare and interpolate the data for a list of symbols this function will download the latest results, 
    store them in a json file for the day if it is the first query
    The Recover the query all previously saved results and appeend it to a single dataframe
    '''

    if (checkRHCredentials() == True):
        if (checkDailyQuery() == True):
            # Check if we already have results for the last day.
            # and query the server to store the new data
            rh = login()   
            queryMultiTickerRHResults(rh)
            queryLatestFunRHResults(rh)    
            
        if (predict == False):
            # df_cap = mergePreviousTickerData(ticker, queryType) 
            df_cap = mergePreviousMultiData(ticker, queryType) 
        else:
            # We need the most recent data to generate the latest results
            rh = login()   
            df_cap = queryLatestTickerRHResults(rh, ticker, queryType)
            df_cap['begins_at'] = pd.to_datetime(df_cap['begins_at'], utc=True)
            df_cap.to_json('temp.json', orient='records', lines=True)
    else:
        df_cap = mergePreviousMultiData(ticker, queryType) 

    df_cap.set_index('begins_at', inplace=True)   
    
    df_cap = df_cap.drop_duplicates()    

    sorted_idx = df_cap.index.sort_values()
    df_cap = df_cap.loc[sorted_idx]
        
    df_cap = simpleAnalysis(df_cap, False)    
    df_cap = appendTickerHistory(df_cap, target_col, PREV_HIST)
    df_cap, marketCap = generateModelIndicators(df_cap, period=1)

    if (upsample != "0"):
        df_cap = interpolateTickerdata(df_cap, "day",  upsample)       
    
    df_cap = addTickerTriggers(df_cap)
        
    # Reverse the dataframe
    # df_reversed = df_cap[::-1].copy()
        
    df_cap.dropna(inplace=True)
    df_cap.to_csv('out.csv', index=True)  
    df_cap.tail()

    return df_cap, marketCap
