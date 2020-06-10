import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl

ticker = "AAPL"
start = datetime.datetime(2012, 1, 1)
end = datetime.datetime(2019, 1, 12)

# get data from start date to end date
apple = web.DataReader(ticker, 'yahoo', start, end)

# get closing value
close_col = apple['Adj Close']

# get the returns of each day
returns = close_col / close_col.shift(1) - 1

returns.plot(label = 'return')
plt.legend()
plt.show()
