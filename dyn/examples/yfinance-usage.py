!pip install yfinance 

import yfinance as yf

ticked = yf.Ticker("^NDX")
hist = ticked.history(period="100d",interval="1d")

#print(hist) #ここをコメントアウトするとデータの詳細が見れる

closes=hist['Close'].values #詳細データから必要なデータをlistで取り出した

#print(closes)  #どんなデータか確認してみよう
