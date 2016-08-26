
# coding: utf-8

# # Capstone project 
# ### Investment and trading project - Predicting stock prices and returns based on machine learning

# ## 1. Data exploration: get data from quandle api. We primarily choose indices across the world. 

# In[ ]:

#get_ipython().magic(u'matplotlib inline')

import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

local_python_path = '/Library/Python/2.7/site-packages'
sys.path.append(local_python_path)


# In[244]:

def getSingleNameStockPrice(symbol, start, end):
    """
    Adjusted close price from quandle.
    """
    import quandl
    mydata = quandl.get(symbol, start_date=start, end_date=end, authtoken="zYuLi6xBbvDYgsQJApiA")
    return mydata["Adjusted Close"]

def getReturns(price):
    """
    Get stock returns
    """
    return price.pct_change()[1:]

def getEsFuturesStockPrice(symbol, start, end):
    """
    Adjusted close price from quandle.
    """
    import quandl
    mydata = quandl.get(symbol, start_date=start, end_date=end, authtoken="zYuLi6xBbvDYgsQJApiA")
    return mydata["Close"]


import quandl
#yahoo_close = getStockPrice("SCF/CME_ES1_FW"", "2001-12-31", "2016-03-01")
#yahoo_close = quandl.get("SCF/CME_ES1_FW", start_date="2001-12-31", end_date="2016-03-01", authtoken="zYuLi6xBbvDYgsQJApiA")
yahoo_close = getEsFuturesStockPrice("GOOG/AMEX_SHY", "2001-12-31", "2016-03-01")

returns = getReturns(yahoo_close)
returns
plt.figure(1)
plt.plot(yahoo_close)

plt.figure(2)
plt.plot(returns)


# In[245]:

yahoo_close


# In[246]:

import statsmodels.graphics.tsaplots as tsplots
tsplots.plot_acf(returns, lags= 20)
plt.show()


# In[247]:

returns[1:10]


# In[248]:

tsplots.plot_pacf(returns, lags= 20)
plt.show()


# In[249]:

from pandas.tools.plotting import bootstrap_plot
bootstrap_plot(returns, size = 50)
plt.show()


# Find out the following facts about the data set
# - Total number of data points
# - Number of positive returns
# - Number of negative returns
# - Average annualized returns
# - Standard deviation
# - Sharpe ratio
# - Cumulative returns

# In[250]:

total_data = len(returns)
positive_returns = sum(x > 0 for x in returns)
negative_returns = sum(x < 0 for x in returns)
annualized_returns = np.mean(returns)*12
std = np.std(returns)*np.sqrt(12)
Sharpe = annualized_returns/std
Cumulative_returns = np.log(yahoo_close[-1]/yahoo_close[0])

print "total_data: {}".format(total_data)
print "positive returns: {}".format(positive_returns)
print "negative returns: {}".format(negative_returns)
print "annualized returns: {}".format(annualized_returns)
print "standard deviation: {}".format(std)
print "Sharpe ratio: {}".format(Sharpe)
print "Cumulative returnsL {}".format(Cumulative_returns)


# ## Preparing the data

# In[251]:

num_train = 0.75 * len(returns)
num_test = len(returns) - num_train
num_all = len(returns)
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(returns[1:-1], returns[2:len(returns)], test_size = float(num_test)/num_all, random_state = 42 )



# ### 1. Use supervised learning algorithms to predict stock returns. Use lagged returns as features. 

# In[252]:

y = returns[1:len(returns)]
features = returns[0:len(returns)-1]
plt.plot(features, y, 'o')


# In[253]:

X_train


# In[267]:

threshold = 0
# Train a supervised model and use the model to predict future stock returns
import time

def train_classifier(clf, X_train, y_train):
    print "Training {}...".format(clf.__class__.__name__)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print "Done!\nTraining time (secs): {:.3f}".format(end - start)

from sklearn.svm import SVC
#clf = SVC()
#from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
#clf = LinearRegression()# MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1) #SVC()
from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()
clf = LinearRegression()
from sklearn.linear_model import LogisticRegression
# Fit model to training data

train_classifier(clf, X_train.reshape(len(X_train), 1), y_train)  # note: using entire training set here
print clf  # you can inspect the learned model by printing it

def predict_returns(clf, X_test):
    return clf.predict(X_test.reshape(len(X_test), 1))


# In[268]:

from sklearn.metrics import mean_squared_error
y_test_pred = predict_returns(clf, X_test)
mse = mean_squared_error(y_test_pred, y_test)
rmse = np.sqrt(mse)


# In[269]:


def sup_trading_strat(predict_returns, y_test):
    total_returns = []
    for i in range(1, len(predict_returns)):
        vol = volatility_estimate(y_test[0:i])
        
        if predict_returns[i-1] >= threshold:
            total_returns.append(y_test[i] * 0.4 / vol)
        elif predict_returns[i-1] < -threshold:
            total_returns.append(-y_test[i] * 0.4 / vol)
            
    avg_ret = np.mean(total_returns) * 12
    avg_std = np.std(total_returns) * np.sqrt(12)
    sharpe = avg_ret/avg_std
    cum = np.cumsum(total_returns)
    return avg_ret, sharpe, cum

avg_ret, sharpe, cum = sup_trading_strat(y_test_pred, y_test)

# calc avg returns, sharpe, cumulative


# In[270]:

print avg_ret
print sharpe
plt.plot(cum)


# ### 2. Use unsupervised learning algorithms to predict the sign of stock returns. Use lagged returns as features

# In[271]:

X = returns[0:-1]
Y = returns[1:len(returns)]
#Y_label = [1 if i >0 else -1 for i in Y]
X_train_un, X_test_un, y_train_un, y_test_un = train_test_split(X, Y, test_size = float(num_test)/num_all, random_state = 42 )
y_train_label = [1 if i >0 else -1 for i in y_train_un]
y_test_label = [1 if i >0 else -1 for i in y_test_un]


# In[272]:

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
clf2 = GradientBoostingClassifier()
train_classifier(clf2, X_train_un.reshape(len(X_train_un), 1), y_train_label)
def predict_return_sign(X_test_un):
    return clf2.predict(X_test_un)

pred_label = predict_return_sign(X_test_un.reshape(len(X_test_un), 1))
score = f1_score(pred_label, y_test_label)
print score


# In[ ]:




# In[273]:

def unsup_trading_strat(predict_returns, y_test):
    total_returns = []
    vols = []
    for i in range(1, len(predict_returns)):
        vol = volatility_estimate(y_test[0:i])
        vols.append(vol)
        if predict_returns[i-1] >= 0:
            total_returns.append(y_test[i] * 0.4 / vol)
        else:
            total_returns.append(-y_test[i] * 0.4 / vol)
            
    avg_ret = np.mean(total_returns) * 12
    avg_std = np.std(total_returns) * np.sqrt(12)
    sharpe = avg_ret/avg_std
    cum = np.cumsum(total_returns)
    return avg_ret, sharpe, cum, total_returns, vols


# In[274]:

un_ret, un_sharpe, un_cum, new_returns, vols = unsup_trading_strat(pred_label, y_test_un)


# In[275]:

print volatility_estimate(y_test[0:10])
print "Average return: {}".format(un_ret)
print "Sharpe: {}".format(un_sharpe)
plt.plot(un_cum)


# In[276]:

vols


# ## Volatility model. 
# ### Use volatility to determine the size of this instrument to maximize the return per risk. 

# In[277]:

def volatility_estimate(r_t):
    var = 0
    r_m = np.mean(r_t)
    
    total_num = len(r_t)
    total_sum = total_num*(total_num+1)/float(2) 
    delta = 0.7
    if total_num == 1:
        return 0.4
    
    for i in range(0, total_num):
        if total_num - i < 30:
            #delta_i = (i+1)/float(total_sum) #(1-delta)* delta_i * 
            var = var + ((1-delta) * total_num +30*delta)/float(30) * (r_t[i] - r_m)**2
        else:
            var = var + delta * (r_t[i] - r_m)**2 
    return np.sqrt(var * 261/float(total_num))


# In[278]:

volatility_estimate(y_test)


# In[279]:

np.std(y_test)*np.sqrt(261)


# In[ ]:



