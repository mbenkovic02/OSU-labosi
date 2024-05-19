from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error
import pandas as pd

data = pd.read_csv('data_C02_emission.csv')
# ucitaj ugradeni podatkovni skup
X, y = datasets.load_diabetes ( return_X_y = True )
# podijeli skup na podatkovni skup za ucenje i poda tkovni skup za testiranje
X_train , X_test , y_train , y_test = train_test_split (X, y, test_size = 0.2, random_state =1)
# min - max skaliranje
sc = MinMaxScaler ()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform (X_test)
ohe = OneHotEncoder()
X_encoded = ohe.fit_transform (data[['Fuel Type']]).toarray()

linearModel = lm.LinearRegression()
linearModel.fit(X_train_n , y_train)

# predikcija izlazne velicine na skupu podataka za testiranje
y_test_p = linearModel . predict (X_test_n)
# evaluacija modela na skupu podataka za testiranje pomocu MAE
MAE = mean_absolute_error (y_test , y_test_p)