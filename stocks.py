import pandas as pd
import yfinance as yf
from datetime import datetime
import plotly.express as px


# Setting start date and end date for the stocks that we want to analyze
start_date = datetime.now() - pd.DateOffset(months=3)
end_date = datetime.now()

tickers = ['AAPL', 'MSFT', 'NFLX', 'GOOG']

df_list = []
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    df_list.append(data)

df = pd.concat(df_list, keys=tickers, names=['Ticker', 'Date'])

#reset index
print(df.head())
print(df.tail())

df = df.reset_index()
print(df.head())
print(df.tail())


#1.Analyze and find the Stock Market Performance for the Last 3 Months and display it as a line chart.
img = px.line(df, x='Date', y='Close', color='Ticker', title='Stocks Performance')
img.show()

#2.Compare the performance of different companies and identify similarities or differences in their stock price movements using the faceted area chart.
img_1 = px.area(df, x='Date', y='Close', color='Ticker', facet_col='Ticker',
                labels={'Date': 'Date', 'Close': 'Closing value', 'Ticker': 'Company'},
                title='Stock Prices')
img_1.show()

# 3.Calculate the 10-day and 20-day moving averages for each company
df['MAverage10'] = df.groupby("Ticker")['Close'].rolling(window=10).mean().reset_index(0, drop=True)
df['MAverage20'] = df.groupby("Ticker")['Close'].rolling(window=20).mean().reset_index(0, drop=True)

# Print the moving averages for each company
for ticker, group in df.groupby('Ticker'):
    print(f'Moving Averages for {ticker}:')
    print(group[['MAverage10', 'MAverage20']])

# Visualize the moving averages for each company
for ticker, group in df.groupby("Ticker"):
    img_3 = px.line(group, x='Date', y=['Close', 'MAverage10', 'MAverage20'],
                     title=f'Moving Averages of Stock Prices for {ticker}')
    img_3.show()

# 4.Calculate the volatility for each company based on the 10-day percentage change in stock prices
df['Volatility'] = df.groupby("Ticker")['Close'].pct_change().rolling(window=10).std().reset_index(0, drop=True)

# Visualize the stock volatility for each company
img_4 = px.line(df, x='Date', y='Volatility', color="Ticker", title='Stock Volatility')
img_4.show()


# 5.Extract data for Apple and Microsoft
apple = df.loc[df['Ticker'] == 'AAPL', ['Date', 'Close']].rename(columns={'Close': 'AAPL'})
microsoft = df.loc[df['Ticker'] == 'MSFT', ['Date', 'Close']].rename(columns={'Close': 'MSFT'})

# Analyze the correlation between the stock prices of Apple and Microsoft
df_correlation = pd.merge(apple, microsoft)
img_5 = px.scatter(df_correlation, x='AAPL', y='MSFT',
                    trendline='ols',
                    title='Correlation between Apple and Microsoft')
img_5.show()

#sentiment analysis
tickers = ['AAPL', 'NFLX']

df_list = []
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    df_list.append(data)

df = pd.concat(df_list, keys=tickers, names=['Ticker', 'Date'])

# Sentiment Analysis (Simple Approach) 
def get_sentiment_score(text):
    # Here, you can implement a basic sentiment analysis logic to determine positive, negative, or neutral sentiment
    # For simplicity, let's assume positive sentiment if the stock price increased, negative if it decreased, and neutral otherwise
    if text > 0:
        return "Positive"
    elif text < 0:
        return "Negative"
    else:
        return "Neutral"

# Calculate sentiment for each company's stock price data
df['Sentiment'] = df.groupby('Ticker')['Close'].pct_change().apply(get_sentiment_score)

print(df)
# Plot the sentiment analysis results
fig = px.line(df.reset_index(), x='Date', y='Close', color='Sentiment', title='Market Sentiment Analysis')
fig.show()

# Calculate the cumulative returns for each company and plot it as a line chart to see the growth over time 
df['Cumulative_Return'] = (1 + df.groupby('Ticker')['Daily_Return'].cumsum()).reset_index(0, drop=True)
img_7 = px.line(df, x='Date', y='Cumulative_Return', color='Ticker',
              title='Cumulative Returns of Different Companies')
img_7.show()

# Analyze the trading volume of each company and visualize it using a bar chart to see if there are any patterns or spikes in volume 
img_8 = px.bar(df, x='Date', y='Volume', color='Ticker', title='Trading Volume of Different Companies')
img_8.show()