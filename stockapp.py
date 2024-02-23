import streamlit as st 
import yfinance as yf
import pandas as pd
import cufflinks as cf 
import datetime 
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# App title
st.markdown('''
# Stock Price and Prediction app
Shown are the stock price data for query companies!

**Credits**
- App built by [Gurpreet kapoor](https://github.com/GURPREETKAPOOR13)
- Built in `Python` using `streamlit`,`yfinance`, `cufflinks`, `pandas`, `prophet`,`plotly` and `datetime`
''')
st.write('---')

# Sidebar
st.sidebar.subheader('Query parameters')
start_date = st.sidebar.date_input("Start date", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date(2024, 1, 1))

# Retrieving tickers data
ticker_list = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt')
tickerSymbol = st.sidebar.selectbox('Stock ticker', ticker_list) # Select ticker symbol
tickerData = yf.Ticker(tickerSymbol) # Get ticker data
tickerDf = tickerData.history(period='1d', start=start_date, end=end_date) #get the historical prices for this ticker

selected_stock = tickerSymbol

# # # Ticker information
try:
  string_logo = '<img src=%s>' % tickerData.info['logo_url']
  st.markdown(string_logo, unsafe_allow_html=True)
except KeyError:
  # Handle missing logo_url gracefully
  print("Logo URL not available")
  # You can display a placeholder image here
  # or skip the image altogether


string_name = tickerData.info['longName']
st.header('**%s**' % string_name)

string_summary = tickerData.info['longBusinessSummary']
st.info(string_summary)

# Ticker data
st.header('**Ticker data**')
st.write(tickerDf)

# Bollinger bands
st.header('**Bollinger Bands**')
qf=cf.QuantFig(tickerDf,title='First Quant Figure',legend='top',name='GS')
qf.add_bollinger_bands()
fig = qf.iplot(asFigure=True)
st.plotly_chart(fig)

####
#st.write('---')
#st.write(tickerData.info)



START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# st.title("Stock prediction")

# stocks = ("GOOG","AAPL","MSFT","GME")
# stocks = ("GOOG","AAPL","MSFT","GME")
# selected_stock = st.text_input("select stock")


n_year = st.sidebar.slider("Year of prediction",1,4)
period= n_year*365

def load_data(stock):
    data = yf.download(stock,START,TODAY)
    data.reset_index(inplace=True)
    return data

data = load_data(selected_stock)
st.subheader("Stock data")
st.write(data.tail())

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=data["Date"],y=data["Open"],name="stock_open"))
fig1.add_trace(go.Scatter(x=data["Date"],y=data["Close"],name="stock_close"))
fig1.layout.update(title_text="time series data",xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)



#predict future

df_train = data[["Date","Close"]]
df_train = df_train.rename(columns={"Date":"ds","Close":"y"})

m=Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)


#plot forecast

st.subheader("Forecast data")
fig2 = plot_plotly(m,forecast)
st.plotly_chart(fig2)


st.markdown("# DISCLAIMER")
st.warning("Please note that the information provided in this app does not replace professional advice from licensed finance professionals and brokers. Due to the inherent risks in stock trading, it is advised that users consult with professionals before making any financial decisions.")


