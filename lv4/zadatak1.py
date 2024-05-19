"""Zadatak 4.5.1 Skripta zadatak_1.py ucitava podatkovni skup iz ˇ data_C02_emission.csv.
Potrebno je izgraditi i vrednovati model koji procjenjuje emisiju C02 plinova na temelju ostalih numerickih ulaznih veli ˇ cina. Detalje oko ovog podatkovnog skupa mogu se prona ˇ ci u 3. ´
laboratorijskoj vježbi.
a) Odaberite željene numericke veli ˇ cine speci ˇ ficiranjem liste s nazivima stupaca. Podijelite
podatke na skup za ucenje i skup za testiranje u omjeru 80%-20%. ˇ
b) Pomocu matplotlib biblioteke i dijagrama raspršenja prikažite ovisnost emisije C02 plinova ´
o jednoj numerickoj veli ˇ cini. Pri tome podatke koji pripadaju skupu za u ˇ cenje ozna ˇ cite ˇ
plavom bojom, a podatke koji pripadaju skupu za testiranje oznacite crvenom bojom. ˇ
c) Izvršite standardizaciju ulaznih velicina skupa za u ˇ cenje. Prikažite histogram vrijednosti ˇ
jedne ulazne velicine prije i nakon skaliranja. Na temelju dobivenih parametara skaliranja ˇ
transformirajte ulazne velicine skupa podataka za testiranje. ˇ
d) Izgradite linearni regresijski modeli. Ispišite u terminal dobivene parametre modela i
povežite ih s izrazom 4.6.
e) Izvršite procjenu izlazne velicine na temelju ulaznih veli ˇ cina skupa za testiranje. Prikažite ˇ
pomocu dijagrama raspršenja odnos izme ´ du stvarnih vrijednosti izlazne veli ¯ cine i procjene ˇ
dobivene modelom.
f) Izvršite vrednovanje modela na nacin da izra ˇ cunate vrijednosti regresijskih metrika na ˇ
skupu podataka za testiranje.
g) Što se dogada s vrijednostima evaluacijskih metrika na testnom skupu kada mijenjate broj ¯
ulaznih velicina?"""

import pandas as pd
import matplotlib . pyplot as plt
from sklearn.base import r2_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn . linear_model as lm
from sklearn . model_selection import train_test_split

data = pd.read_csv("lv4\data_C02_emission.csv")

#a)

input_variables = ['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)']
output = 'CO2 Emissions (g/km)'

X, y = data[input_variables], data[output]

X_train , X_test , y_train , y_test = train_test_split (X , y , test_size = 0.2 , random_state =1 )

#b)

plt.scatter(X_train['Engine Size (L)'], y_train, c='blue')
plt.scatter(X_test['Engine Size (L)'], y_test, c='red')
plt.show()

#c)

sc = MinMaxScaler ()

X_train_n = sc . fit_transform ( X_train )   # min max scaler vraca numpy a ne dataframe
X_test_n = sc . transform ( X_test )

plt.hist(X_train['Engine Size (L)'], label='Original')
plt.hist(X_train_n[:, 0], label='Scaled')  # Accessing the first column of X_train_n (Engine Size (L))
plt.show()

#d)

linearModel = lm . LinearRegression ()
linearModel . fit ( X_train_n , y_train )

print("Coefficients:", linearModel.coef_)
print("Intercept:", linearModel.intercept_)


#e)


y_test_p = linearModel . predict ( X_test_n )

plt.scatter(x= y_test, y= y_test_p)
plt.xlabel ("Stvarne vrijednosti")
plt.ylabel ("Predviđene vrijednosti")
plt.title ("Odnos stvarnih i predviđenih vrijednosti")
plt.show ()

#f)

MAE = mean_absolute_error(y_test , y_test_p)
MSE = mean_squared_error(y_test , y_test_p)
MAPE = mean_absolute_percentage_error(y_test, y_test_p)
RMSE = mean_squared_error(y_test, y_test_p, squared=False)
R_TWO_SCORE = r2_score(y_test, y_test_p)

print(f"MAE: {MAE}, MSE: {MSE}, MAPE: {MAPE}, RMSE: {RMSE}, R2 SCORE: {R_TWO_SCORE}")