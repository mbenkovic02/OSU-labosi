"""Datoteka titanic.csv sadrži podatke o putnicima broda Titanic, koji je potonuo
1912. godine. Upoznajte se s datasetom i dodajte programski kod u skriptu pomo´ cu kojeg možete
odgovoriti na sljede´ ca pitanja:
a) Za koliko žena postoje podatci u ovom skupu podataka?
b) Koliki postotak osoba nije preživio potonu´ ce broda?
c) Pomo´ cu stupˇ castog dijagrama prikažite postotke preživjelih muškaraca (zelena boja) i žena
(žuta boja). Dodajte nazive osi i naziv dijagrama. Komentirajte korelaciju spola i postotka
preživljavanja.
d) Kolika je prosjeˇcna dob svih preživjelih žena, a kolika je prosjeˇcna dob svih preživjelih
muškaraca?
e) Koliko godina ima najstariji preživjeli muškarac u svakoj od klasa? Komentirajte.
Zadatak0.0.5 Datoteka titanic.csv sadrži podatke o putnicima broda Titanic, koji je potonuo
1912. godine. Upoznajte se s datasetom. Uˇ citajte dane podatke. Podijelite ih na ulazne podatke X
predstavljene stupcima Pclass, Sex, Fare i Embarked i izlazne podatke y predstavljene stupcem
Survived. Podijelite podatke na skup za uˇcenje i skup za testiranje modela u omjeru 70:30.
Izbacite izostale i null vrijednosti. Skalirajte podatke. Dodajte programski kod u skriptu pomo´ cu
kojeg možete odgovoriti na sljede´ ca pitanja:
a) Izradite algoritam KNN na skupu podataka za uˇ cenje (uz K=5). Vizualizirajte podatkovne
primjere i granicu odluke."""



from keras import models
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from tensorflow import keras
from keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder


data_pd = pd.read_csv("2/titanic.csv")

data_pd.info()

#a)

data_women = data_pd[data_pd['Sex'] == 'female']

print(f"Broj zena:", len(data_women))

#b)

data_deceased = data_pd[data_pd['Survived'] == 0]

print(f"Percentage deceased:", len(data_deceased)/len(data_pd)*100)

#c)
males=data_pd[data_pd['Sex']=='male']
survived_men = data_pd[(data_pd['Sex'] == 'male') & (data_pd['Survived'] == 1)]
percentage_men=(len(survived_men)/len(males)*100)

females=data_pd[data_pd['Sex']=='female']
survived_women = data_pd[(data_pd['Sex'] == 'female') & (data_pd['Survived'] == 1)]
percentage_women=(len(survived_women)/len(females)*100)

plt.bar(['Males', 'Females'], [percentage_men, percentage_women], color=['green', 'yellow'])

plt.xlabel("Gender")
plt.ylabel("Percentage Survived")
plt.title("Percentage of men and women survived")
plt.legend()
plt.show()

#d)

print(f"Prosjecna dob prezivjelih muskaraca:", survived_men['Age'].mean())
print(f"Prosjecna dob prezivjelih zena:", survived_women['Age'].mean())

#e)

data_pdd= data_pd[(data_pd['Pclass']==2)&(data_pd['Sex']=='male')&(data_pd['Survived']==1)]
print(f"AAAAAAAAAAAAAAAA", data_pdd['Age'].max())

print(data_pd[(data_pd['Pclass']==1)&(data_pd['Sex']=='male')&(data_pd['Survived']==1)].sort_values(by=['Age'],ascending=False).head(1))
print(data_pd[(data_pd['Pclass']==2)&(data_pd['Sex']=='male')&(data_pd['Survived']==1)].sort_values(by=['Age'],ascending=False).head(1))
print(data_pd[(data_pd['Pclass']==3)&(data_pd['Sex']=='male')&(data_pd['Survived']==1)].sort_values(by=['Age'],ascending=False).head(1))




""" Datoteka titanic.csv sadrži podatke o putnicima broda Titanic, koji je potonuo
 1912. godine. Upoznajte se s datasetom. Uˇ citajte dane podatke. Podijelite ih na ulazne podatke X
 predstavljene stupcima Pclass, Sex, Fare i Embarked i izlazne podatke y predstavljene stupcem
 Survived. Podijelite podatke na skup za uˇcenje i skup za testiranje modela u omjeru 70:30.
 Izbacite izostale i null vrijednosti. Skalirajte podatke. Dodajte programski kod u skriptu pomo´ cu
 kojeg možete odgovoriti na sljede´ ca pitanja:
 a) Izradite algoritam KNN na skupu podataka za uˇ cenje (uz K=5). Vizualizirajte podatkovne
 primjere i granicu odluke.
4
 b) Izraˇ cunajte toˇ cnost klasifikacije na skupu podataka za uˇ cenje i skupu podataka za testiranje.
 Komentirajte dobivene rezultate.
 c) Pomo´cu unakrsne validacije odredite optimalnu vrijednost hiperparametra K algoritma
 KNN.
 d) Izraˇ cunajte toˇ cnost klasifikacije na skupu podataka za uˇ cenje i skupu podataka za testiranje
 za dobiveni K. Usporedite dobivene rezultate s rezultatima kada je K=5."""





data_selected = data_pd[['Pclass','Sex','Fare','Embarked','Survived']]

data_selected.info()

print(f"Null redci:\n", data_selected.isnull().sum())
print(f"Duplicirani redci:\n", data_selected.duplicated().sum())

data_selected= data_selected.dropna(how='any', axis =0)
data_selected= data_selected.drop_duplicates()

data_selected.info()



input_variables = ['Pclass','Sex','Fare','Embarked']
output = 'Survived'

X, y = data_selected[input_variables], data_selected[output]

X_train , X_test , y_train , y_test = train_test_split (X , y , test_size = 0.3 , random_state =1 )





ohe = OneHotEncoder ()

X_encoded_train = ohe.fit_transform(X_train[['Sex', 'Embarked']]).toarray()
X_encoded_test = ohe.fit_transform(X_test[['Sex', 'Embarked']]).toarray()




print(X_encoded_train)

sc = StandardScaler()
X_train_n = sc.fit_transform(X_encoded_train)
X_test_n = sc.transform(X_encoded_test)



print(X_train_n)

"""
X_train_str = X_train.iloc[:, [1, 3]]
X_train_num = X_train.iloc[: , [0, 2]]

X_test_str = X_test.iloc[:, [1, 3]]
X_test_num = X_test.iloc[: , [0, 2]]


ohe = OneHotEncoder ()

X_encoded_train_str = ohe.fit_transform(X_train_str).toarray()
X_encoded_test_str = ohe.fit_transform(X_test_str).toarray()




sc = StandardScaler()
X_scaled_train_num = sc.fit_transform(X_train_num)
X_scaled_test_num = sc.transform(X_test_num)




X_train_combined = np.hstack((X_encoded_train_str, X_scaled_train_num))
X_test_combined = np.hstack((X_encoded_test_str, X_scaled_test_num))
"""

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


#a)

KNN_model = KNeighborsClassifier(n_neighbors = 5)
KNN_model.fit(X_train_n, y_train)

y_test_p = KNN_model.predict(X_test_n)
y_train_p = KNN_model.predict(X_train_n)




#b)


print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

#c)


KNN_model = KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(X_train_n, y_train)
y_test_p_KNN = KNN_model.predict(X_test_n)
y_train_p_KNN = KNN_model.predict(X_train_n)

model = KNeighborsClassifier()
scores = cross_val_score(KNN_model, X_train_n, y_train, cv=5)
print(scores)

array = np.arange(1, 101)
param_grid = {'n_neighbors':array}
knn_gscv = GridSearchCV(KNN_model, param_grid , cv=5, scoring ='accuracy', n_jobs =-1)
knn_gscv.fit(X_train_n, y_train)
print(knn_gscv.best_params_)
print(knn_gscv.best_score_)
print(knn_gscv.cv_results_)
print(knn_gscv.best_params_)


#d)


KNN_model = KNeighborsClassifier(n_neighbors = 5)
KNN_model.fit(X_train_n, y_train)

y_test_a = KNN_model.predict(X_test_n)
y_train_a = KNN_model.predict(X_train_n)

print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_a))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_a))))




KNN_model = KNeighborsClassifier(n_neighbors = 20)
KNN_model.fit(X_train_n, y_train)

y_test_b = KNN_model.predict(X_test_n)
y_train_b = KNN_model.predict(X_train_n)

print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_b))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_b))))


"""
 Datoteka titanic.csv sadrži podatke o putnicima broda Titanic, koji je potonuo
 1912. godine. Upoznajte se s datasetom. Uˇ citajte dane podatke. Podijelite ih na ulazne podatke X
 predstavljene stupcima Pclass, Sex, Fare i Embarked i izlazne podatke y predstavljene stupcem
 Survived. Podijelite podatke na skup za uˇcenje i skup za testiranje modela u omjeru 75:25.
 Izbacite izostale i null vrijednosti. Skalirajte podatke. Dodajte programski kod u skriptu pomo´ cu
 kojeg možete odgovoriti na sljede´ ca pitanja:
 a) Izgradite neuronsku mrežu sa sljede´ cim karakteristikama:- model oˇ cekuje ulazne podatke X- prvi skriveni sloj ima 12 neurona i koristi relu aktivacijsku funkciju- drugi skriveni sloj ima 8 neurona i koristi relu aktivacijsku funkciju- tre´ ci skriveni sloj ima 4 neurona i koristi relu aktivacijsku funkciju- izlazni sloj ima jedan neuron i koristi sigmoid aktivacijsku funkciju.
 Ispišite informacije o mreži u terminal.
 b) Podesite proces treniranja mreže sa sljede´ cim parametrima:- loss argument: binary_crossentropy- optimizer: adam- metrika: accuracy.
 c) Pokrenite uˇcenje mreže sa proizvoljnim brojem epoha (pokušajte sa 100) i veliˇcinom
 batch-a 5.
 d) Pohranite model na tvrdi disk te preostale zadatke izvršite na temelju uˇ citanog modela.
 e) Izvršite evaluaciju mreže na testnom skupu podataka.
 f) Izvršite predikciju mreže na skupu podataka za testiranje. Prikažite matricu zabune za skup
 podataka za testiranje. Komentirajte dobivene rezultate i predložite kako biste ih poboljšali,
 ako je potrebno
"""








data_selected = data_pd[['Pclass','Sex','Fare','Embarked','Survived']]

data_selected.info()

print(f"Null redci:\n", data_selected.isnull().sum())
print(f"Duplicirani redci:\n", data_selected.duplicated().sum())

data_selected= data_selected.dropna(how='any', axis =0)
data_selected= data_selected.drop_duplicates()

data_selected.info()



input_variables = ['Pclass','Sex','Fare','Embarked']
output = 'Survived'

X, y = data_selected[input_variables], data_selected[output]

X_train , X_test , y_train , y_test = train_test_split (X , y , test_size = 0.25 , random_state =1 )





ohe = OneHotEncoder ()

X_encoded_train = ohe.fit_transform(X_train[['Sex', 'Embarked']]).toarray()
X_encoded_test = ohe.fit_transform(X_test[['Sex', 'Embarked']]).toarray()




sc = StandardScaler()
X_train_n = sc.fit_transform(X_encoded_train)
X_test_n = sc.transform(X_encoded_test)

#a)

model = keras . Sequential ()
model . add ( layers . Input ( shape=(419,)))
model . add ( layers . Dense (12 , activation ="relu") )
model . add ( layers . Dense (8 , activation ="relu") )
model . add ( layers . Dense (4 , activation ="relu") )
model . add ( layers . Dense (1 , activation ="sigmoid") )
model . summary ()

#b)

model . compile ( loss ="binary_crossentropy" , optimizer ="adam", metrics = ["accuracy", ])

#c)

history = model . fit ( X_train_n , y_train , batch_size = 5 , epochs = 100 , validation_split = 0.1)

#d)

model.save('ti_model.keras')

#e i f)

model = models.load_model('zadatak_1_model.keras')

predictions = model . predict (X_test_n)

score = model . evaluate ( X_test_n , y_test , verbose =0 )

disp = ConfusionMatrixDisplay(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1)))
disp.plot()
plt.show()