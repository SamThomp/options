from OP_Predict_Config import *
# Functions for handling the data manipulation


def mergeFundamentals(ticker):
    """
    Read all of the fundamentals data files are in this project and concatenate them
    into a single dataframe
    """
    import os

    sub_dir = "data/fundamentals"
    list_of_files = os.listdir(os.getcwd() + "/" + sub_dir) #list of files in the current directory
    each_file = ""
    tickerHist = []

    for each_file in list_of_files:
        if each_file.startswith(ticker):  #since its all type str you can simply use startswith
            # print (each_file)
            saved_ticker_hist = pd.read_json(sub_dir + each_file)
            tickerHist.append(saved_ticker_hist)

    df = pd.concat(tickerHist).drop_duplicates()
    return df

def checkDailyQuery():
    """
    Read all of the trading data files are in this project
    """
    import os

    today =  str(datetime.today().strftime('%Y_%m_%d'))
    print(today)
    sub_dir = "data/long/"
    list_of_files = os.listdir(os.getcwd() + "/" + sub_dir) #list of files in the current directory
    each_file = ""

    for each_file in list_of_files:
        if each_file.endswith(today+".json") == True:  #since its all type str you can simply use startswith
            print("Server already queried for today, training data is stocked")
            print(each_file)
            return False
            
    return True

def mergePreviousTickerData(ticker, period="short"):
    """
    Read all of the trading data files are in this project
    """
    import os

    if (period == "short"):
        sub_dir = "data/short/"
    else:
        sub_dir = "data/long/"
    list_of_files = os.listdir(os.getcwd() + "/" + sub_dir) #list of files in the current directory
    each_file = ""

    df = pd.DataFrame([])
    for each_file in list_of_files:
        if each_file.startswith(ticker):  #since its all type str you can simply use startswith
            # print(each_file)
            df = pd.concat([df, pd.read_json(sub_dir + each_file)])

    print("loading files from:" + sub_dir)
    print(df.shape)
    return df



def visualize_loss(history, title):
    """
    We can visualize the loss with the function below. After one point, the loss stops
    decreasing.
    """
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def mergePreviousMultiData(ticker, period="short"):
    """
    Read all of the trading data files are in this project
    """
    import os

    if (period == "short"):
        sub_dir = "data/short/"
    else:
        sub_dir = "data/long/"
    list_of_files = os.listdir(os.getcwd() + "/" + sub_dir) #list of files in the current directory
    each_file = ""

    df = pd.DataFrame([])
    for each_file in list_of_files:
        if each_file.startswith('multi'):  #since its all type str you can simply use startswith
            df = pd.concat([df, pd.read_json(sub_dir + each_file)])

    
    print("loading files from:" + sub_dir)
    print(df.shape)
    return df.loc[df['symbol'] == ticker]