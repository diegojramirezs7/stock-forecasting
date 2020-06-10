import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import pandas_datareader.data as web
import datetime
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl

start = datetime.datetime(2012, 1, 1)
end = datetime.datetime(2019, 1, 12)

dfcomp = web.DataReader(['AAPL', 'FB', 'GOOGL', 'AMZN', 'MSFT'],'yahoo',start=start,end=end)['Adj Close']
composite_returns = dfcomp.pct_change()
correlation = composite_returns.corr()

# visualize correlations among all stocks selected from DataReader
scatter_matrix(composite_returns, diagonal='kde', figsize=(10, 10))

#adjust size of matplotlib
mpl.rc('figure', figsize=(8, 7))

#adjust style of matplotlib
style.use('ggplot')

plt.legend()
plt.show()
