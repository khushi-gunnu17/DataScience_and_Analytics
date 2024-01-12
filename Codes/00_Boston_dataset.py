from sklearn import datasets
boston = datasets.load_boston()
# type(data) = sklearn.utils.bunch
# data = to get the entire data 
X = boston.data
Y = boston.target

X.shape
# (506, 13)

import pandas as pd
df = pd.DataFrame(X)

print(boston.feature_names)
df.columns = boston.feature_names
df.describe()

boston.DESCR

# splitting the data 
from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y)
#  determines the size of data in cols and rows
print(X_train.shape)       
print(X_test.shape)
print(Y_train.shape)       
print(Y_test.shape)

# get the algorithm into the code 
# linear regression classifier
from sklearn.linear_model import LinearRegression
# this gives us the algorithm object that we can use to train and test the data
alg1 = LinearRegression()
alg1.fit(X_train, Y_train)
# now, it's predicting the output from the testing data
Y_pred = alg1.predict(X_test)
# compare Y_pred and Y_test

# plot the graph to compare Y_predict and Y_test
import matplotlib.pyplot as plt 
plt.scatter(Y_test, Y_pred)
# align the axis
plt.axis([0, 40, 0, 40])
plt.show()