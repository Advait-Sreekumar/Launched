import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("SALES.txt", delim_whitespace=True, header=None)
df.columns=['Sales', 'Advertising']

print(df.head())

print(df.info())

print(df.describe())

X = df['Sales'].values
y = df['Advertising'].values


plt.scatter(X, y, color = 'blue', label='Scatter Plot')
plt.title('Relationship between Sales and Advertising')
plt.xlabel('Sales')
plt.ylabel('Advertising')
plt.legend(loc=4)
plt.show()

print(X.shape)
print(y.shape)

X = X.reshape(-1,1)
y = y.reshape(-1,1)
print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()


# Train the model using training data sets
lm.fit(X_train,y_train)


# Predict on the test data
y_pred=lm.predict(X_test)



#Predicting graph
a = lm.coef_
b = lm.intercept_,
print("Estimated model slope, a:" , a)
print("Estimated model intercept, b:" , b)

lm.predict(X)[0:5]
print(str(lm.predict([[26]])))


#RMSE
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE value: {:.5f}".format(rmse))


#R2
from sklearn.metrics import r2_score
print ("R2 Score value: {:.5f}".format(r2_score(y_test, y_pred)))

plt.scatter(X, y, color = 'red', label='Scatter Plot')
plt.plot(X_test, y_pred, color = 'black', linewidth=3, label = 'Regression Line')
plt.title('Relationship between Sales and Advertising')
plt.xlabel('Sales')
plt.ylabel('Advertising')
plt.legend(loc=4)
plt.show()

#Residual erros
plt.scatter(lm.predict(X_train), lm.predict(X_train) - y_train, color = 'red', label = 'Train data')
plt.scatter(lm.predict(X_test), lm.predict(X_test) - y_test, color = 'blue', label = 'Test data')
plt.hlines(xmin = 0, xmax = 50, y = 0, linewidth = 3)
plt.title('Residual errors')
plt.legend(loc = 4)
plt.show()


print("Training set score: {:.5f}".format(lm.score(X_train,y_train)))

print("Test set score: {:.5f}".format(lm.score(X_test,y_test)))