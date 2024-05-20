""" Datoteka winequality-red.csv sadrži kemijska mjerenja crnih vina i njihovu
kvalitetu. Upoznajte se s datasetom. Više informacija nalazi se u datoteci winequality.names.
Dodajte programski kod u skriptu pomo´ cu kojeg možete odgovoriti na sljede´ ca pitanja:
a) Na koliko je crnih vina provedeno mjerenje?
b) Prikažite stupˇ castim dijagramima (na jednoj slici) ovisnost kvalitete vina o alkoholnoj ja
kosti. Dodajte naziv dijagrama i nazive osi. Interpretirajte rezultate prikazane dijagramom.
c) Koliki broj uzoraka vina ima kvalitetu manju od 5, a koliki ima 5 i ve´ cu?
d) Izraˇ cunajte i prikažite korelaciju svih veliˇ cina dostupnih u datasetu. Interpretirajte rezultate."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn . linear_model as lm
from keras import models
#a)

data_pd = pd.read_csv("3\winequality-red.csv", delimiter=';')

data_pd.info()

print("Broj mjerenih vina:", len(data_pd))

#b)

data_grouped = data_pd.groupby('quality')['alcohol'].mean()
data_grouped.plot(kind = 'bar', xlabel='Quality', ylabel='Alcohol', title='Alcohol to Quality relation')
plt.show()

#c)

data_less = data_pd[data_pd['quality']<5]
data_more = data_pd[(data_pd['quality'] > 5) | (data_pd['quality'] == 5)]

print("Kvaliteta manja od 5: ", len(data_less))
print("Kvaliteta veca ili jednaka 5: ", len(data_more))

#d)
pd.options.display.max_columns = None

print(data_pd.corr(numeric_only=True))
matrix=data_pd.corr(numeric_only=True)
sn.heatmap(matrix, annot= True)
plt.show()



"""
1 Datoteka winequality-red.csv sadrži kemijska mjerenja crnih vina i njihovu
 kvalitetu. Upoznajte se s datasetom. Više informacija nalazi se u datoteci winequality.names.
 Uˇcitajte dane podatke. Podijelite skup na ulazne podatke X i izlazne podatke y predstavljene
 kvalitetom vina. Zamijenite sve vrijednosti kvalitete vina manje od 5 s 0, a one koje imaju
 vrijednost 5 ili ve´ cu s 1 kako biste dobili dvije izlazne klase. Podijelite podatke na skup za uˇ cenje
 i skup za testiranje modela u omjeru 80:20. Standardizirajte podatke. Dodajte programski kod u
 skriptu pomo´ cu kojeg možete odgovoriti na sljede´ ca pitanja:
 a) Izgradite linearni regresijski model. Ispišite parametre modela.
 b) Izvršite procjenu izlazne veliˇ cine na temelju ulaznih veliˇ cina skupa za testiranje. Prikažite
 pomo´ cu dijagrama raspršenja odnos izme¯ du stvarnih vrijednosti izlazne veliˇ cine i procjene
 dobivene modelom. Interpretirajte dobivene rezultate.
8
 c) Izvršite vrednovanje modela na naˇ cin da izraˇ cunate vrijednosti regresijskih metrika (RMSE,
 MAE, MAPE iR2) na skupu podataka za testiranje. Interpretirajte dobivene rezultate.
"""

input= data_pd.drop(['quality'],axis=1)

output = data_pd['quality']


"""
for index, value in output.items():
    if value < 5:
        data_pd.loc[index, 'quality'] = 0
    else:
        data_pd.loc[index, 'quality'] = 1

"""
pd.options.display.max_rows = None

output[output < 5] = 0
output[output>= 5] = 1

X=input
y=output

X_train , X_test , y_train , y_test = train_test_split (X , y , test_size = 0.2 , random_state =1 )

sc = MinMaxScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)

#a)

linearModel = lm . LinearRegression ()
linearModel . fit ( X_train_n , y_train )

print("Coefficients:", linearModel.coef_)
print("Intercept:", linearModel.intercept_)

#b)

y_test_p = linearModel . predict ( X_test_n )


plt.scatter(x= y_test, y= y_test_p)
plt.xlabel ("Stvarne vrijednosti")
plt.ylabel ("Predviđene vrijednosti")
plt.title ("Odnos stvarnih i predviđenih vrijednosti")
plt.show ()



RMSE = mean_squared_error(y_test, y_test_p, squared=False)
MAE = mean_absolute_error(y_test , y_test_p)
MAPE = mean_absolute_percentage_error(y_test, y_test_p)


print(f"RMSE: {RMSE}, MAE: {MAE}, MAPE: {MAPE}")


"""
 Datoteka winequality-red.csv sadrži kemijska mjerenja crnih vina i njihovu
 kvalitetu. Upoznajte se s datasetom. Više informacija nalazi se u datoteci winequality.names.
 Uˇcitajte dane podatke. Podijelite skup na ulazne podatke X i izlazne podatke y predstavljene
 kvalitetom vina. Zamijenite sve vrijednosti kvalitete vina manje od 5 s 0, a one koje imaju
 vrijednost 5 ili ve´ cu s 1 kako biste dobili dvije izlazne klase. Podijelite podatke na skup za uˇ cenje
 i skup za testiranje modela u omjeru 80:20. Pripremite podatke za uˇ cenje.
 a) Izgradite neuronsku mrežu sa sljede´ cim karakteristikama:- model oˇ cekuje ulazne podatke X- prvi skriveni sloj ima 20 neurona i koristi relu aktivacijsku funkciju- drugi skriveni sloj ima 12 neurona i koristi relu aktivacijsku funkciju- tre´ ci skriveni sloj ima 4 neurona i koristi relu aktivacijsku funkciju- izlazni sloj ima jedan neuron i koristi sigmoid aktivacijsku funkciju.
 Ispišite informacije o mreži u terminal.
 b) Podesite proces treniranja mreže sa sljede´ cim parametrima:- loss argument: binary_crossentropy- optimizer: adam- metrika: accuracy.
 c) Pokrenite uˇcenje mreže sa proizvoljnim brojem epoha (pokušajte s 1000) i proizvoljnom
 veliˇ cinom batch-a (pokušajte s 50).
 d) Pohranite model na tvrdi disk te preostale zadatke izvršite na temelju uˇ citanog modela.
 e) Izvršite evaluaciju mreže na testnom skupu podataka.
 f) Izvršite predikciju mreže na skupu podataka za testiranje. Prikažite matricu zabune za skup
 podataka za testiranje. Interpretirajte dobivene rezultate"""

input= data_pd.drop(['quality'],axis=1)

output = data_pd['quality']


"""
for index, value in output.items():
    if value < 5:
        data_pd.loc[index, 'quality'] = 0
    else:
        data_pd.loc[index, 'quality'] = 1

"""
pd.options.display.max_rows = None

output[output < 5] = 0
output[output>= 5] = 1

X=input
y=output

X_train , X_test , y_train , y_test = train_test_split (X , y , test_size = 0.2 , random_state =1 )

sc = MinMaxScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)

model = keras . Sequential ()
model . add ( layers . Input ( X_train_n.shape[1] ))
model . add ( layers . Dense (20 , activation ="relu") )
model . add ( layers . Dense (12 , activation ="relu") )
model . add ( layers . Dense (4 , activation ="relu") )
model . add ( layers . Dense (1 , activation ="sigmoid") )
model . summary ()


model . compile ( loss ="binary_crossentropy" , optimizer ="adam", metrics = ["accuracy", ])
history = model . fit ( X_train_n , y_train , batch_size = 50 , epochs = 1000 , validation_split = 0.1)

model.save('wi_model.keras')

model = models.load_model('zadatak_1_model.keras')

predictions = model . predict (X_test_n)

score = model . evaluate ( X_test_n , X_train_n , verbose =0 )

disp = ConfusionMatrixDisplay(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1)))
disp.plot()
plt.show()