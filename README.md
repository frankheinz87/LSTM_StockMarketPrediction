# LSTM_StockMarketPrediction
This project is the first personal project of my boot.dev backend developer learning path.

I will follow the instructions of this guide closely (all credits for this part go to the author):  
[LSTM Python Stock Market](https://www.datacamp.com/tutorial/lstm-python-stock-market)

Following that i will imlement a simple feature to evaluate the accuracy of the predictions.

How to use:
* activate venv: source .venv/bin/activate
* sudo apt-get install python3-tk
* arguments of main.py:
    1. data source: either "kaggle" or "alphavantage"
    2. type: either "Stocks" or "ETFs"
    3. the actual stock:
        + if kaggle: name of the .txt file in the corresponding folder (so either Stocks or ETFs and e.g. "hpq.us.txt")
        + if alphavantage: Ticker of the corresponding stock e.g. AAL 
