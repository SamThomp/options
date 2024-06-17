
import json
import pandas as pd
from datetime import datetime
import os

from OP_Predict_Config import *


def checkRHCredentials():
    try:
        print(RH_KEY)
        return True
    except NameError:
        print("Robhinhood Credentials, are not defined")
    return False

def login():



    totp = pyotp.TOTP(RH_KEY)
    print("Current OTP:", totp.now())
    print("Current OTP:", totp)
    # Here I am setting store_session=False so no pickle file is used.
    login = rh.login(RH_E, RH_P, store_session=False, mfa_code=totp.now())
    # In the login dictionary, you will see that 'detail' is 
    # 'logged in with brand new authentication code.' to show that I am not using a pickle file.
    print(login)
    
    return rh

def queryLatestFunRHResults(rh):
    tickers = ['AAPL','NDAQ', 'SPY', 'XLF', 'LCID', 'VEA', 'MSFT', 'VOO', 'AMD', 'AMZN', 'META','SMCI','CRM', 'GOOG', 'COST', 'V', 'SBUX','KLAC','AXP','ANET','GE','INTC']
    
    # External Calls    
    queryResponse = rh.get_fundamentals(tickers)
    filename = str('data/fundamentals/' + 'multi' + '_' + str(datetime.today().strftime('%Y_%m_%d')) +'.json')
    with open(filename, 'w') as f:
        json.dump(queryResponse, f)

    df_fun = pd.read_json(filename) 
    
    return df_fun


def queryLatestTickerRHResults(rh, symbol, queryType):
    """
    Use the Robinhood API to query the latest ticker data from the site
    """    
    if (queryType == 'long'):
        interval = "hour"
        span = "3month"
    else:
        interval = "5minute"
        span = "week"

    queryResponse = rh.get_stock_historicals(symbol, interval, span)
    latest_ticker_results = pd.DataFrame(queryResponse)
    
    return latest_ticker_results



def queryMultiTickerRHResults(rh):
    """
    Use the Robinhood API to query the latest ticker data from the site
    """
    tickers = ['AAPL','NDAQ', 'SPY', 'XLF', 'LCID', 'VEA', 'MSFT', 'VOO', 'AMD', 'AMZN', 'META','SMCI','CRM', 'GOOG', 'COST', 'V', 'SBUX','KLAC','AXP','ANET','GE','INTC']

    def saveHistory(rh, tickers, interval, span, dir):        
        counter = 0
        
        # External Calls        
        queryResponse = rh.get_stock_historicals(tickers, interval, span)
        
        filename = str(dir + 'multi' + '_' + interval  + '_'  + span + '_' + str(datetime.today().strftime('%Y_%m_%d')) +'.json')
        with open(filename, 'w') as f:
            json.dump(queryResponse, f)

    saveHistory(rh, tickers, "5minute", "week", "data/short/")
    saveHistory(rh, tickers, "hour", "3month", "data/long/")


def queryOpenPositions(rh):
    positions = rh.get_open_stock_positions()

    for position in positions:    
        if ('symbol' in position):
            if ('average_buy_price' in position):
                if ('quantity' in position):
                    print(f"{position['symbol']} { round(Decimal(position['quantity']), 3) } at: {position['average_buy_price']}")

    return positions


def checkLifetimeGains():

    '''
    Robinhood includes dividends as part of your net gain. This script removes
    dividends from net gain to figure out how much your stocks/options have paid
    off.

    Note: load_portfolio_profile() contains some other useful breakdowns of equity.
    Print profileData and see what other values you can play around with.
    See: https://github.com/jmfernandes/robin_stocks/blob/master/examples/robinhood%20examples/get_accurate_gains.py

    '''
    profileData = rh.load_portfolio_profile()
    allTransactions = rh.get_bank_transfers()
    cardTransactions= rh.get_card_transactions()

    deposits = sum(float(x['amount']) for x in allTransactions if (x['direction'] == 'deposit') and (x['state'] == 'completed'))
    withdrawals = sum(float(x['amount']) for x in allTransactions if (x['direction'] == 'withdraw') and (x['state'] == 'completed'))
    debits = sum(float(x['amount']['amount']) for x in cardTransactions if (x['direction'] == 'debit' and (x['transaction_type'] == 'settled')))
    reversal_fees = sum(float(x['fees']) for x in allTransactions if (x['direction'] == 'deposit') and (x['state'] == 'reversed'))

    money_invested = deposits + reversal_fees - (withdrawals - debits)
    dividends = rh.get_total_dividends()
    if (money_invested == 0):
        percentDividend = 0.0
    else:
        percentDividend = dividends/money_invested*100

    equity = float(profileData['extended_hours_equity'])
    totalGainMinusDividends = equity - dividends - money_invested
    if (money_invested == 0):
        percentGain = 0.0
    else:
        percentGain = totalGainMinusDividends/money_invested*100

    print("The total money invested is {:.2f}".format(money_invested))
    print("The total equity is {:.2f}".format(equity))
    print("The net worth has increased {:0.2}% due to dividends that amount to {:0.2f}".format(percentDividend, dividends))
    print("The net worth has increased {:0.3}% due to other gains that amount to {:0.2f}".format(percentGain, totalGainMinusDividends))

    return [money_invested, equity]





def getCallOptions(rh, stock, strike, minutesToTrack: int = 60, PrintInterval: int = 10, date = "2024-06-10", optionType: str = "call"):

    '''
    This is an example script that will print out options data every 10 seconds for 1 minute.
    It also saves the data to a txt file. The txt file is saved in the same directory as this code.

    from utilities import optionsWriter
    import importlib
    importlib.reload(optionsWriter)

    result = optionsWriter.getCallOptions("AAPL", 168)
    '''

    #!!! fill out the specific option information
    # strike = strikePrice
    # date = "2024-04-20"
    # stock = ["LCID", "AAPL", "NDAQ "]
    # optionType = "call" #or "put"
    #!!!

    # File saving variables in minutes
    endTime = t.time() + 60 * minutesToTrack
    fileName = "options.txt"
    writeType = "w" #or enter "a" to have it continuously append every time script is run
    #

    os.chdir(os.path.dirname(__file__))
    path = os.getcwd()
    filename = os.path.join(path,fileName)
    fileStream = open(filename, mode=writeType)

    while t.time() < endTime:
        time = str(datetime.now())
        #Both write and print the data so that you can view it as it runs.
        fileStream.write("\n")
        fileStream.write(time)
        print(time)
        #Get the data
        # instrument_data = r.get_option_instrument_data(stock,date,strike,optionType)
        instrument_data = rh.find_options_by_expiration(stock, date)
        
        print(instrument_data)
        market_data = rh.get_option_market_data(stock, date, strike, optionType)

        fileStream.write("\n")
        fileStream.write("{} Instrument Data {}".format("="*30,"="*30))
        print("{} Instrument Data {}".format("="*30,"="*30))
        # instrument_data is a dictionary, and the key/value pairs can be accessed with .items()
        for key, value in instrument_data.items():
            fileStream.write("\n")
            fileStream.write("key: {:<25} value: {}".format(key,value))
            print("key: {:<25} value: {}".format(key,value))

        fileStream.write("\n")
        fileStream.write("{} Market Data {}".format("="*30,"="*30))
        print("{} Market Data {}".format("="*30,"="*30))

        for key, value in market_data[0].items():
            fileStream.write("\n")
            fileStream.write("key: {:<25} value: {}".format(key,value))
            print("key: {:<25} value: {}".format(key,value))

        t.sleep(PrintInterval)

    # make sure to close the file stream when you are done with it.
    fileStream.close()


def printOpenPositions(minutesToTrack: int = 10, PrintInterval: int = 60):
    """
    Print all of the open positions in robinhood
    minutesToTrack - number of minutes to track the open positions
    PrintInterval - How often to query and print the resuls
    """

    minutesToTrack = 1 #in minutes
    PrintInterval = 30 #in seconds
    endTime = t.time() + 60 * minutesToTrack
    # Get live positions from my account
    positions = rh.get_open_stock_positions()

    while t.time() < endTime:
        time = str(datetime.datetime.now())
        print(time)    
        # Loop through all of the open account positions, and check if its time to unload
        for position in positions:    
            if ('symbol' in position):
                if ('average_buy_price' in position):
                    if ('quantity' in position):
                        print(f"{position['symbol']} { round(Decimal(position['quantity']), 3) } at: {position['average_buy_price']}")                                            

        t.sleep(PrintInterval)


def checkPortfolioPerformance():
    '''
    This is an example script that will show you how to check the performance of your open positions.
    '''
    # Query your positions
    positions = rh.get_open_stock_positions()

    # Get Ticker symbols
    tickers = [rh.get_symbol_by_url(item["instrument"]) for item in positions]

    # Get your quantities
    quantities = [float(item["quantity"]) for item in positions]

    # Query previous close price for each stock ticker
    prevClose = rh.get_quotes(tickers, "previous_close")

    # Query last trading price for each stock ticker
    lastPrice = rh.get_quotes(tickers, "last_trade_price")

    # Calculate the profit per share
    profitPerShare = [float(lastPrice[i]) - float(prevClose[i]) for i in range(len(tickers))]

    # Calculate the percent change for each stock ticker
    percentChange = [ 100.0 * profitPerShare[i] / float(prevClose[i]) for i in range(len(tickers)) ]

    # Calcualte your profit for each stock ticker
    profit = [profitPerShare[i] * quantities[i] for i in range(len(tickers))]

    # Combine into list of lists, for sorting
    tickersPerf = list(zip(profit, percentChange, tickers))

    tickersPerf.sort(reverse=True)

    print ("My Positions Performance:")
    print ("Ticker | DailyGain | PercentChange")
    for item in tickersPerf:
        print ("%s %f$ %f%%" % (item[2], item[0], item[1]))

    print ("Net Gain:", sum(profit))

    return profit