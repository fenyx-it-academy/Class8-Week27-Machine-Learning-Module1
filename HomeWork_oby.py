import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

data=pd.read_csv('weatherAUS.csv')
df = pd.read_csv('weatherAUS.csv')

#dealing with nan values
df = df.fillna(df.mean())

#clear data
selected_columns = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustDir', 'WindGustSpeed', 'WindSpeed9am',
                    'WindSpeed3pm', 'Pressure9am', 'Pressure3pm', 'Humidity9am', 'Humidity3pm']

X = df[selected_columns]


#catagorical to numeric
print(X['Location'].nunique())

cat_columns=['Location','WindGustDir']
for i in cat_columns:
    
# Turn my column into a dummy value
    dummy = pd.get_dummies(X[i])   
    X = pd.concat([X, dummy], axis=1).drop(i, axis=1)


y=X['MaxTemp']
X.drop('MaxTemp', axis=1,inplace=True)


from sklearn.model_selection import train_test_split,cross_val_score
x_train, x_test,y_train,y_test = train_test_split(X,y,test_size=0.33, random_state=0)

#1-Linear Regression

from sklearn.linear_model import LinearRegression, Ridge , Lasso
lin_reg = LinearRegression()
lin_reg.fit(x_train,y_train)

#scores
y_hat=lin_reg.predict(x_test)
from sklearn.metrics import mean_squared_error , mean_absolute_error

print('--------------------')
print('Linear regression')
mae = mean_absolute_error(y_true= y_test , y_pred= y_hat)
print('Mean Absolute Error (MAE) with test data' ,mae)

mse = mean_squared_error(y_true=y_test , y_pred= y_hat)
print("Mean Squared Error (MSE) with test data" , mse)



#test with all values

lin_reg2 = LinearRegression()
lin_reg2.fit(X,y)

y_hat=lin_reg2.predict(X)

mae = mean_absolute_error(y_true= y , y_pred=y_hat )
print('Mean Absolute Error (MAE) with all data' ,mae)

mse = mean_squared_error(y_true=y , y_pred= y_hat)
print("Mean Squared Error (MSE) with all data" , mse)

print('R2 score with all data',r2_score(y, lin_reg.predict(X)))



#2 Polyynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2)

X_poly = poly_features.fit_transform(X)

poly_regression = LinearRegression()
poly_regression.fit(X_poly, y)

# Predicting values
X_pred_poly = poly_features.transform(X)
y_hat_poly = poly_regression.predict(X_pred_poly)
print('--------------------')
print('Polynomial Regrassion')
mae = mean_absolute_error(y_true= y , y_pred=y_hat )
print('Mean Absolute Error (MAE) with all data' ,mae)

mse = mean_squared_error(y_true=y , y_pred= y_hat)
print("Mean Squared Error (MSE) with all data" , mse)

print('R2 score with all data',r2_score(y, lin_reg.predict(X)))

#3 lasso and ridge
lasso = Lasso()
ridge = Ridge()
lasso_fit = lasso.fit(x_train , y_train)
ridge_fit = ridge.fit(x_train , y_train)

print("RIDGE -> Score on Training Set ", ridge.score(x_train,y_train).round(4) , "\tScore on testing set ", ridge.score(x_test,y_test).round(4))
print("LASSO -> Score on Training Set ", lasso.score(x_train,y_train).round(4) , "\tScore on testing set ", lasso.score(x_test,y_test).round(4))


cross_val_score(ridge, x_train, y_train, cv = 10, scoring = 'r2')

print("cross validation score MLR " , cross_val_score(lin_reg, x_train, y_train, cv = 10, scoring = 'r2').mean().round(6))
print("cross validation score RIDGE " ,  cross_val_score(ridge, x_train, y_train, cv = 10, scoring = 'r2').mean().round(6))
print("cross validation score LASSO " , cross_val_score(lasso, x_train, y_train, cv = 10, scoring = 'r2').mean().round(6))


#Scaling data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaler_y = MinMaxScaler(feature_range=(0,1))
X_normalized = scaler.fit_transform(X)
Y_normalized = scaler_y.fit_transform(np.array(y).reshape(len(y),1))

X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X_normalized, Y_normalized, test_size=0.2, shuffle = True, random_state = 15)

mlr_n = LinearRegression()
ridge_n = Ridge()
lasso_n = Lasso()

mlr_n_fit = mlr_n.fit(X_train_n , y_train_n)
ridge_n_fit = ridge_n.fit(X_train_n , y_train_n)
lasso_n_fit = lasso_n.fit(X_train_n , y_train_n)

print("MLR -> Score on Normalized Training Set ", mlr_n.score(X_train_n,y_train_n).round(4) , "\tScore on Normalized testing set ", mlr_n.score(X_test_n,y_test_n).round(4))
print("RIDGE -> Score on Normalized Training Set ", ridge_n.score(X_train_n,y_train_n).round(4) , "\tScore on Normalized testing set ", ridge_n.score(X_test_n,y_test_n).round(4))
print("LASSO -> Score on Normalized Training Set ", lasso_n.score(X_train_n,y_train_n).round(4) , "\tScore on testing set ", lasso_n.score(X_test_n,y_test_n).round(4))

print("cross validation score MLR (normalized) " , cross_val_score(mlr_n, X_train_n, y_train_n, cv = 10, scoring = 'r2').mean().round(4))
print("cross validation score RIDGE (normalized) " ,  cross_val_score(ridge_n, X_train_n, y_train_n, cv = 10, scoring = 'r2').mean().round(4))
print("cross validation score LASSO (normalized) " , cross_val_score(lasso_n, X_train_n, y_train_n, cv = 10, scoring = 'r2').mean().round(4))

























