"""Zadatak 4.5.2 Na temelju rješenja prethodnog zadatka izradite model koji koristi i kategoricku ˇ
varijable „Fuel Type“ kao ulaznu velicinu. Pri tome koristite 1-od-K kodiranje kategori ˇ ckih ˇ
velicina. Radi jednostavnosti nemojte skalirati ulazne veli ˇ cine. Komentirajte dobivene rezultate. ˇ
Kolika je maksimalna pogreška u procjeni emisije C02 plinova u g/km? O kojem se modelu
vozila radi?"""

import pandas as pd
import matplotlib . pyplot as plt

from sklearn.metrics import max_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import sklearn . linear_model as lm
from sklearn . model_selection import train_test_split


data = pd.read_csv("lv4\data_C02_emission.csv")

input_variables = ['Fuel Type','Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)']
output = 'CO2 Emissions (g/km)'

X, y = data[input_variables], data[output]


X_train , X_test , y_train , y_test = train_test_split (X , y , test_size = 0.2 , random_state =1 )


ohe = OneHotEncoder ()

X_encoded_train = ohe.fit_transform(X_train[['Fuel Type']]).toarray()
X_encoded_test = ohe.fit_transform(X_test[['Fuel Type']]).toarray()


linearModel = lm . LinearRegression ()
linearModel . fit ( X_encoded_train , y_train )

y_test_p = linearModel . predict ( X_encoded_test )

ME = max_error(y_test, y_test_p)
print(f"Max Error: {ME}")

# Find index of maximum error
max_error_index = abs(y_test - y_test_p).argmax()


# Get corresponding entry in test set (vehicle model)
vehicle_model = data.iloc[max_error_index]             #koristim data jer u x skupu nemam ime vozila

print(f"Vehicle model with maximum error: {vehicle_model}")