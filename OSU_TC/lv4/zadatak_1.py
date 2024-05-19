from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

# Zadatak pod a)
data = pd.read_csv('data_C02_emission.csv')

input_variables = ['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)']
output = 'CO2 Emissions (g/km)'

X = data[input_variables]
y = data[output]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=1)

# Zadatak pod b)
plt.scatter(X_train['Engine Size (L)'], y_train, c='Red')
plt.scatter(X_test['Engine Size (L)'], y_test, c='Blue')
plt.xlabel('Engine Size (L)')
plt.ylabel('CO2 Emissions (g/km)')
plt.title('Emissions compared to engine size')
plt.show()

# Zadatak pod c)
plt.hist(X_train['Engine Size (L)'])
plt.show()

sc = MinMaxScaler()
X_train_n = sc.fit_transform(X_train)
plt.hist
plt.hist(X_train_n[:, 0])
plt.show()

X_test_n = sc.transform(X_test)

# Zadatak pod d)
linearModel = lm.LinearRegression()
linearModel.fit(X_train_n, y_train)

print(linearModel.coef_)

# Zadatak pod e)
y_test_p = linearModel.predict(X_test_n)
plt.scatter(y_test, y_test_p)
plt.title("Real values compared to predicted values")
plt.xlabel("Real values")
plt.ylabel("Predicted values")
plt.show()

# Zadatak pod f)
MAE = mean_absolute_error(y_test , y_test_p)
MSE = mean_squared_error(y_test , y_test_p)
MAPE = mean_absolute_percentage_error(y_test, y_test_p)
RMSE = mean_squared_error(y_test, y_test_p, squared=False)
R_TWO_SCORE = r2_score(y_test, y_test_p)

print(f"MAE: {MAE}, MSE: {MSE}, MAPE: {MAPE}, RMSE: {RMSE}, R2 SCORE: {R_TWO_SCORE}")


