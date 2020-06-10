import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
import math
from sklearn.model_selection import train_test_split

ticker = "FB"
start = datetime.datetime(2012, 1, 1)
end = datetime.datetime(2019, 1, 12)

# use Apple stocks from 2012-2019
df = web.DataReader(ticker, 'yahoo', start, end)

#get closing value
close_col = df['Adj Close']

#calculate moving average for the last 100 windows (days) and take avg of each window's moving avg
moving_avg = close_col.rolling(window=100).mean()

#create the dataframe to be used
# we include the high-low percentage and the percentage change features + adj close and volume that were already given
dfreg = df.loc[:,['Adj Close','Volume']]
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

# fill the missing values, pad method simply carries last valid observation forward
dfreg.fillna(method='pad', inplace=True)

# We want to separate 1 percent of the data to forecast
# how many entries equal 1 % of the total data entries
forecast_out = int(math.ceil(0.2 * len(dfreg)))
# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'

# add the label col to df with adj close values except for the last 1 %
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))

# standardize dataset to be used with linear regression
# it will have properties of normal distribution with mean = 0 and std dev = 1
X = preprocessing.scale(X)


# last 1 % of X 
X_recent = X[-forecast_out:]
# first 99 % of X
X = X[:-forecast_out]

# Separate label and identify it as y
y = np.array(dfreg['label'])
# first 99 % of y
y = y[:-forecast_out]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Linear Regression
# n_jobs param -1 indicates that all available processors should be used
linear_classifier = LinearRegression(n_jobs=-1)
linear_classifier.fit(X_train, y_train)
prediction = linear_classifier.predict(X_test)
# print("intercept: ",linear_classifier.intercept_)
# print("coefficient: ",linear_classifier.coef_)

#quadratic discriminant analysis
quadratic_classifier = make_pipeline(PolynomialFeatures(2), Ridge())
quadratic_classifier.fit(X_train, y_train)

#quadratic discriminant analysis using 3 polynomial features
quadratic_classifier3 = make_pipeline(PolynomialFeatures(3), Ridge())
quadratic_classifier3.fit(X_train, y_train)

# KNN Regression
knn_classifier = KNeighborsRegressor(n_neighbors=2)
knn_classifier.fit(X_train, y_train)


linear_score = linear_classifier.score(X_test, y_test)
quad_score = quadratic_classifier.score(X_test,y_test)
quad3_score = quadratic_classifier3.score(X_test,y_test)
knn_score = knn_classifier.score(X_test, y_test)


print("%.5f"%linear_score)
print("%.5f"%quad_score)
print("%.5f"%quad3_score)
print("%.5f"%knn_score)


#adjust size of matplotlib
mpl.rc('figure', figsize=(8, 7))

#adjust style of matplotlib
# style.use('ggplot')
# prediction.plot(label="prediction")
# y_test.plo(label = "expected")

style.use('ggplot')
close_col.plot(label=ticker)
moving_avg.plot(label='moving average')
plt.legend()
plt.show()








