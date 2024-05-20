#import bibilioteka
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm
import seaborn as sn
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from tensorflow import keras
from keras import layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

data = pd.read_csv('3/winequality-red.csv', delimiter=';')

##################################################
#1. zadatak
##################################################


#a)
print(f"Broj vina na koji je mjerenje: {len(data)}")
#b)
plt.figure()
data['alcohol'].plot(kind='hist', bins=20)
plt.title('Distribucija alkohole jakosti')
plt.xlabel('Alkohol')
plt.ylabel('Broj uzoraka')
plt.show()
#c)
kvaliteta_manja_od_6 = data[data['quality'] < 6]
kvaliteta_veca_jednako_6 = data[data['quality'] >= 6]
print(f"Kvaliteta manja od 6: {len(kvaliteta_manja_od_6)}")
print(f"Kvaliteta vea jednako 6: {len(kvaliteta_veca_jednako_6)}")
#d)
print(data.corr(numeric_only=True))
matrix = data.corr()
sn.heatmap(matrix, annot= True)
plt.show()


##################################################
#2. zadatak
##################################################
X = data.iloc[:, :-1]
y = data['quality']

y[y < 6] = 0
y[y >= 6] = 1

X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.2, random_state =1)
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)
#a)
linearModel = lm.LinearRegression()
linearModel.fit(X_train_n, y_train)
print("Koeficijenti modela")
print(linearModel.coef_)
print("intercept:")
print(linearModel.intercept_)
#b)
y_test_p = linearModel.predict(X_test_n)
plt.scatter(y_test, y_test_p)
plt.xlabel('Stvarne vrijednosti')
plt.ylabel('Predviđene vrijednosti')
plt.show()
#c)
MSE = mean_squared_error(y_test, y_test_p)
RMSE = math.sqrt(MSE)
print(f"RMSE: {RMSE}")

MAE = mean_absolute_error(y_test, y_test_p)
print(f"MAE: {MAE}")

MAPE = mean_absolute_percentage_error(y_test, y_test_p)
print(f"MAPE: {MAPE}")
R2 = r2_score(y_test, y_test_p)
print(f"R2: {R2}")


##################################################
#3. zadatak
##################################################

#a)
model = keras.Sequential()
model.add(layers.Dense(22, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(12, activation='relu'))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

#b)
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy",]
)

#c)
batch_size = 800
epochs = 50
history = model.fit(
    X_train,
    y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.1
)

#d)
model.save('model.keras')
#e)
predictions = model.predict(X_test)
score = model.evaluate(X_test, y_test, verbose =0 )
print("evaluacija", score)
#f)
y_test_p_binary = np.where(y_test_p > 0.5, 1, 0)
cm = confusion_matrix(y_test, y_test_p_binary)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()








