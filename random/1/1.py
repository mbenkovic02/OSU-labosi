""" Datoteka pima-indians-diabetes.csv sadrži mjerenja provedena u svrhu
 otkrivanja dijabetesa, pri ˇcemu se u devetom stupcu nalazi klasa 0 (nema dijabetes) ili klasa 1
 (ima dijabetes). Uˇcitajte dane podatke u obliku numpy polja data. Dodajte programski kod u
 skriptu pomo´ cu kojeg možete odgovoriti na sljede´ ca pitanja:
 a) Na temelju veliˇ cine numpy polja data, na koliko osoba su izvršena mjerenja?
 b) Postoje li izostale ili duplicirane vrijednosti u stupcima s mjerenjima dobi i indeksa tjelesne
 mase (BMI)? Obrišite ih ako postoje. Koliko je sada uzoraka mjerenja preostalo?
 c) Prikažite odnos dobi i indeksa tjelesne mase (BMI) osobe pomo´cu scatter dijagrama.
 Dodajte naziv dijagrama i nazive osi s pripadaju´cim mjernim jedinicama. Komentirajte
 odnos dobi i BMI prikazan dijagramom.
 d) Izraˇ cunajte i ispišite u terminal minimalnu, maksimalnu i srednju vrijednost indeksa tjelesne
 mase (BMI) u ovom podatkovnom skupu.
 e) Ponovite zadatak pod d), ali posebno za osobe kojima je dijagnosticiran dijabetes i za one
 kojima nije. Kolikom je broju ljudi dijagonosticiran dijabetes? Komentirajte dobivene
 vrijednosti."""
from keras import layers
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from keras import models


data_np = np.loadtxt("1\pima-indians-diabetes.csv", delimiter=",", dtype="str")

#a)

print(f"Broj mjerenja:", len(data_np))

#b)

column_names=['Times Pregnant', 'Plasma Glucose', 'Diastolic BP', 'Triceps', 'Serum', 'BMI', 'DPF', 'Age', 'Class']


data_pd = pd.read_csv("1\pima-indians-diabetes.csv", names= column_names, skiprows=9)

# 1. Number of times pregnant
# 2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 3. Diastolic blood pressure (mm Hg)
# 4. Triceps skin fold thickness (mm)
# 5. 2-Hour serum insulin (mu U/ml)
# 6. Body mass index (weight in kg/(height in m)^2)
# 7. Diabetes pedigree function
# 8. Age (years)
# 9. Class variable (0 or 1)

data_pd.info()

print(f"Null redci:\n", data_pd.isnull().sum())
print(f"Duplicirani redci:\n", data_pd.duplicated().sum())

data_pd.dropna(axis =0)
data_pd.drop_duplicates()

data_pd = data_pd[data_pd.iloc[:, 5] != 0 ]


data_pd.info()


print(f"Preostale vrijednosti:", len(data_pd))

#c)

plt.scatter(x= data_pd['Age'], y= data_pd['BMI'])
plt.xlabel('Age')
plt.ylabel('BMI')
plt.title('Age and BMI ratio')
plt.show()

#Vecina ljudi se krece u BMI rasponu od 20 do 40 bez obzira na godine

#d)

print(f"Min BMI:", data_pd['BMI'].min())
print(f"Max BMI:", data_pd['BMI'].max())
print(f"Avg BMI:", data_pd['BMI'].mean())

#e)

data_diabetic = data_pd[data_pd['Class'] == 1]
data_nondiabetic = data_pd[data_pd['Class'] == 0]

print("Values for diabetics:")
print(f"Min BMI:", data_diabetic['BMI'].min())
print(f"Max BMI:", data_diabetic['BMI'].max())
print(f"Avg BMI:", data_diabetic['BMI'].mean())


print("Values for non-diabetics:")
print(f"Min BMI:", data_nondiabetic['BMI'].min())
print(f"Max BMI:", data_nondiabetic['BMI'].max())
print(f"Avg BMI:", data_nondiabetic['BMI'].mean())


#####################################################
"""Datoteka pima-indians-diabetes.csv sadrži mjerenja provedena u svrhu
 otkrivanja dijabetesa, pri ˇ cemu se u devetom stupcu nalazi izlazna veliˇ cina, predstavljena klasom
 0 (nema dijabetes) ili klasom 1 (ima dijabetes).
 Uˇcitajte dane podatke u obliku numpy polja data. Podijelite ih na ulazne podatke X i izlazne
2
 podatke y. Podijelite podatke na skup za uˇcenje i skup za testiranje modela u omjeru 80:20.
 Dodajte programski kod u skriptu pomo´ cu kojeg možete odgovoriti na sljede´ ca pitanja:
 a) Izgradite model logistiˇ cke regresije pomo´ cu scikit-learn biblioteke na temelju skupa poda
taka za uˇ cenje.
 b) Provedite klasifikaciju skupa podataka za testiranje pomo´ cu izgra¯ denog modela logistiˇ cke
 regresije.
 c) Izraˇ cunajte i prikažite matricu zabune na testnim podacima. Komentirajte dobivene rezul
tate.
 d) Izraˇcunajte toˇcnost, preciznost i odziv na skupu podataka za testiranje. Komentirajte
 dobivene rezultate."""

X = data_pd.drop(columns=['Class']).to_numpy()  #svi osim izlaznog
y = data_pd['Class'].copy().to_numpy() 



X_train , X_test , y_train , y_test = train_test_split (X , y , test_size = 0.2 , random_state =1 )


#a)

logisticRegression = LogisticRegression(max_iter=300)
logisticRegression.fit(X_train,y_train)

#b)

y_test_p = logisticRegression.predict( X_test )

#c)

cm = confusion_matrix ( y_test , y_test_p )
print (" Matrica zabune : " , cm )
disp = ConfusionMatrixDisplay ( confusion_matrix ( y_test , y_test_p ) )
disp . plot ()
plt . show ()

#d)

print(f'Tocnost: {accuracy_score(y_test, y_test_p)}')
print(f'Preciznost: {precision_score(y_test, y_test_p)}')
print(f'Odziv: {recall_score(y_test, y_test_p)}')

"""
Matrica zabune prikazuje da je model točno predvidio 89 osoba koje nemaju dijabetes (pravi negativi) i 36 osoba koje imaju dijabetes (pravi pozitivi). Međutim, 
model je pogrešno predvidio 18 osoba koje imaju dijabetes kao da ih nemaju (lažni negativi) i 11 osoba koje nemaju 
dijabetes kao da ga imaju (lažni pozitivi).

Točnost modela iznosi 0.783, što znači da je model točno klasificirao 78.3% ukupnih primjera.
Preciznost modela iznosi 0.674, što znači da je od svih primjera koje je model predvidio kao pozitivne, njih 67.4% zaista pozitivni. 
Odziv (senzitivnost) modela iznosi 0.633, što znači da je model uspio prepoznati 63.3% svih stvarnih pozitivnih primjera
"""
#############################################################
"""Datoteka pima-indians-diabetes.csv sadrži mjerenja provedena u svrhu
 otkrivanja dijabetesa, pri ˇ cemu je prvih 8 stupaca ulazna veliˇcina, a u devetom stupcu se nalazi
 izlazna veliˇ cina: klasa 0 (nema dijabetes) ili klasa 1 (ima dijabetes).
 Uˇ citajte dane podatke. Podijelite ih na ulazne podatke X i izlazne podatke y. Podijelite podatke
 na skup za uˇ cenje i skup za testiranje modela u omjeru 80:20.
 a) Izgradite neuronsku mrežu sa sljede´ cim karakteristikama:- model oˇ cekuje ulazne podatke s 8 varijabli- prvi skriveni sloj ima 12 neurona i koristi relu aktivacijsku funkciju- drugi skriveni sloj ima 8 neurona i koristi relu aktivacijsku funkciju- izlasni sloj ima jedan neuron i koristi sigmoid aktivacijsku funkciju.
 Ispišite informacije o mreži u terminal.
 b) Podesite proces treniranja mreže sa sljede´ cim parametrima:- loss argument: cross entropy- optimizer: adam- metrika: accuracy.
 c) Pokrenite uˇcenje mreže sa proizvoljnim brojem epoha (pokušajte sa 150) i veliˇcinom
 batch-a 10.
 d) Pohranite model na tvrdi disk te preostale zadatke izvršite na temelju uˇ citanog modela.
 e) Izvršite evaluaciju mreže na testnom skupu podataka.
 f) Izvršite predikciju mreže na skupu podataka za testiranje. Prikažite matricu zabune za skup
 podataka za testiranje. Komentirajte dobivene rezultate"""


X = data_pd.drop(columns=['Class']).to_numpy()  #svi osim izlaznog
y = data_pd['Class'].copy().to_numpy() 



X_train , X_test , y_train , y_test = train_test_split (X , y , test_size = 0.2 , random_state =1 )

#a)

model = keras.Sequential()
model.add(layers.Input(shape=(8,)))
model.add(layers.Flatten())       #za error:  Arguments `target` and `output` must have the same rank (ndim). Received: target.shape=(None, 10), output.shape=(None, 28, 28, 10)
model . add ( layers . Dense (12 , activation ="relu") )
model . add ( layers . Dense (8 , activation ="relu") )
model . add ( layers . Dense (1 , activation ="sigmoid") )
model . summary ()

#b)

model.compile(loss="binary_crossentropy",
optimizer="adam",
metrics=["accuracy",])

#c)

history = model.fit(X_train,
y_train,
batch_size = 10,
epochs = 150, 
validation_split = 0.1)


predictions = model . predict (X_test)


#d)

model.save('zadatak_1_model.keras')

#e)

model = models.load_model('zadatak_1_model.keras')


score = model . evaluate ( X_test , y_test , verbose =0 )

#f)

disp = ConfusionMatrixDisplay(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1)))
disp.plot()
plt.show()