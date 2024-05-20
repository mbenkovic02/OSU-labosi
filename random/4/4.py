""" Iris Dataset sastoji se od informacija o laticama i ˇ cašicama tri razliˇ cita cvijeta
 irisa (Setosa, Versicolour i Virginica). Dostupan je u sklopu bibilioteke scikitlearn:
 from sklearn import datasets
 iris = datasets.load_iris()
 Upoznajte se s datasetom i dodajte programski kod u skriptu pomo´ cu kojeg možete odgovoriti na
 sljede´ ca pitanja:
 a) Prikažite odnos duljine latice i ˇcašice svih pripadnika klase Virginica pomo´cu scatter
 dijagrama plavom bojom. Dodajte naziv dijagrama, nazive osi te legendu. Komentirajte
 prikazani dijagram.
 b) Nanovom scatter dijagramu prikažite odnos duljine i širine latice klase Setosa crvenom
 bojom, te duljine i širine ˇ cašice Versicolor zelenom bojom. Dodajte naziv dijagrama, nazive
 osi te legendu. Komentirajte prikazani dijagram.
 c) Pomo´ cu stupˇ castog dijagrama prikažite prosjeˇ cnu vrijednost duljine ˇ cašice za sve tri klase
 cvijeta. Dodajte naziv dijagrama i nazive osi. Komentirajte prikazani dijagram.
 d) Koliko jedinki pripadnika klase Setosa ima ve´ cu duljinu ˇ cašice od prosjeˇ cne širine ˇ cašice
 te klase?
"""

#https://www.jcchouinard.com/sklearn-datasets-iris/

from matplotlib import pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
iris = datasets.load_iris()

data_pd = pd.DataFrame(iris.data, columns=iris.feature_names)



data_pd['target'] = iris.target

target_names = {
    0:'setosa',
    1:'versicolor', 
    2:'virginica'
}
 
data_pd['target_names'] = data_pd['target'].map(target_names)
data_pd.head()

data_pd.info()

print(data_pd)


#a)

data_virginica = data_pd[data_pd['target'] == 2]


plt.scatter(x=data_virginica['sepal length (cm)'], y=data_virginica['sepal width (cm)'])
plt . xlabel ("Length")
plt . ylabel ("Width")
plt . title ( "Length-Width relation")
plt . show ()

#b)

data_setosa = data_pd[data_pd['target'] == 0]
data_versicolor = data_pd[data_pd['target'] == 1]

plt.scatter(x=data_setosa['sepal length (cm)'], y=data_setosa['sepal width (cm)'], color= 'red')
plt.scatter(x=data_versicolor['sepal length (cm)'], y=data_versicolor['sepal width (cm)'], color= 'green')
plt . xlabel ("Length")
plt . ylabel ("Width")
plt . title ( "Length-Width relation")
plt . show ()

#c)

data = data_pd.groupby('target_names')['sepal length (cm)'].mean()

data.plot(kind ='bar', xlabel='Flowers', ylabel='sepal length (cm)', title='Average length comparison')
plt.show()


#d)

average_width = data_setosa['sepal width (cm)'].mean()
filtered= data_setosa[data_setosa['sepal length (cm)'] > average_width]

print("Broj Setosa s vecom duljinom od prosjecne sirine:", len(filtered))


"""
 Iris Dataset sastoji se od informacija o laticama i ˇ cašicama tri razliˇ cita cvijeta
 irisa (Setosa, Versicolour i Virginica). Dostupan je u sklopu bibilioteke scikitlearn:
10
 from sklearn import datasets
 iris = datasets.load_iris()
 Upoznajte se s datasetom. Pripremite podatke za uˇcenje. Dodajte programski kod u skriptu
 pomo´ cu kojeg možete odgovoriti na sljede´ ca pitanja:
 a) Prona¯ dite optimalni broj klastera K (od 1 do 10) za klasifikaciju cvijeta irisa algoritmom K
 srednjih vrijednosti koriste´ci lakat metodu te ju prikažite grafiˇcki (dodati naslov i nazive
 osi).
 b) Primijenite algoritam K srednjih vrijednosti koji ´ ce prona´ ci grupe u podatcima. Koristite
 vrijednost K dobivenu u prethodnom zadatku.
 c) Dijagramom raspršenja prikažite dobivene klastere te ih obojite razliˇ citim bojama: Setosa
zelenom, Virginica- plavom i Vesrsicolor- crvenom. Centroide prikazati crnom bojom.
 Dodajte nazive osi, naslov dijagrama i legendu. Komentirajte dijagram!
 d) Izraˇ cunajte preciznost klasifikacije.
"""

data_new = data_pd.drop(columns=['target_names'])

data_new.info()

wcss = []  # within-cluster sums of squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(data_new)
    wcss.append(kmeans.inertia_)


plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Lakat metoda')
plt.xlabel('Broj klastera')
plt.ylabel('WCSS')
plt.show()



kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(data_new)

data_new['cluster'] = clusters

data_pd['cluster'] = clusters

# Boje za različite klastere
colors = {0: 'red', 1: 'green', 2: 'blue'}
cluster_colors = data_pd['cluster'].map(colors)

# Raspršeni dijagram
plt.figure(figsize=(10, 6))
plt.scatter(data_pd['petal length (cm)'], data_pd['petal width (cm)'], c=cluster_colors, label='Clusters', alpha=0.6)
plt.title('Klasteri cvijeta irisa')
plt.xlabel('Duljina latica (cm)')
plt.ylabel('Širina latica (cm)')
plt.legend()
plt.show()


# Mapiranje klastera na originalne klase
def map_cluster_to_target(cluster_label):
    if cluster_label == 0:
        return 2  # virginica
    elif cluster_label == 1:
        return 0  # setosa
    else:
        return 1  # versicolor

data_pd['cluster_mapped'] = data_pd['cluster'].map(map_cluster_to_target)



# Izračunavanje preciznosti
accuracy = accuracy_score(data_pd['target'], data_pd['cluster_mapped'])
conf_matrix = confusion_matrix(data_pd['target'], data_pd['cluster_mapped'])

print(f'Preciznost klasifikacije: {accuracy}')
print(f'Matrici konfuzije:\n{conf_matrix}')


"""
 Iris Dataset sastoji se od informacija o laticama i ˇ cašicama tri razliˇ cita cvijeta
 irisa (Setosa, Versicolour i Virginica). Dostupan je u sklopu bibilioteke scikitlearn:
 from sklearn import datasets
 iris = datasets.load_iris()
 Upoznajte se s datasetom. Podijelite ga na ulazne podatke X i izlazne podatke y predstavljene
 klasom cvijeta. Pripremite podatke za uˇ cenje neuronske mreže (kategoriˇ cke veliˇ cine, skaliranje...).
 Podijelite podatke na skup za uˇcenje i skup za testiranje modela u omjeru 75:25. Dodajte
 programski kod u skriptu pomo´ cu kojeg možete odgovoriti na sljede´ ca pitanja:
 a) Izgradite neuronsku mrežu sa sljede´ cim karakteristikama:- model oˇ cekuje ulazne podatke X- prvi skriveni sloj ima 10 neurona i koristi relu aktivacijsku funkciju- drugi skriveni sloj ima 7 neurona i koristi relu aktivacijsku funkciju- tre´ ci skriveni sloj ima 5 neurona i koristi relu aktivacijsku funkciju- izlazni sloj ima 3 neurona i koristi softmax aktivacijsku funkciju.-izme¯ du prvog i drugog te drugog i tre´ceg sloja dodajte Dropout sloj s 30% izbaˇcenih
 neurona
 Ispišite informacije o mreži u terminal.
 b) Podesite proces treniranja mreže sa sljede´ cim parametrima:- loss argument: categorical_crossentropy- optimizer: adam- metrika: accuracy.
 c) Pokrenite uˇcenje mreže sa proizvoljnim brojem epoha (pokušajte sa 500) i veliˇcinom
 batch-a 7.
 d) Pohranite model na tvrdi disk te preostale zadatke izvršite na temelju uˇ citanog modela.
 e) Izvršite evaluaciju mreže na testnom skupu podataka.
 f) Izvršite predikciju mreže na skupu podataka za testiranje. Prikažite matricu zabune za skup
 podataka za testiranje. Komentirajte dobivene rezultate i predložite kako biste ih poboljšali,
 ako je potrebno
"""

data_neuron = data_pd.drop(columns=['target_names'])

input = ['sepal length (cm)',  'sepal width (cm)', 'petal length (cm)',  'petal width (cm)']

output = 'target'

X=data_neuron[input]
y=data_neuron[output]

X_train , X_test , y_train , y_test = train_test_split (X , y , test_size = 0.25 , random_state =1 )

scaler = StandardScaler()
X_train_n = scaler.fit_transform(X_train)
X_test_n = scaler.transform(X_test)

y_train_categorical = utils.to_categorical(y_train)
y_test_categorical = utils.to_categorical(y_test)

print(X_train_n.shape)


model = keras . Sequential ()
model . add ( layers . Input ( shape=(4,) ))
model . add ( layers . Dense (10 , activation ="relu") )
model.add(layers.Dropout(0.3))
model . add ( layers . Dense (7 , activation ="relu") )
model.add(layers.Dropout(0.3))
model . add ( layers . Dense (5 , activation ="relu") )
model . add ( layers . Dense (3 , activation ="softmax") )
model . summary ()


model . compile ( loss ="categorical_crossentropy" , optimizer ="adam", metrics = ["accuracy", ])

batch_size = 7
epochs = 500
history = model . fit ( X_train_n , y_train_categorical , batch_size = batch_size , epochs = epochs , validation_split = 0.1)


model.save('flo_model.keras')

model = models.load_model('zadatak_1_model.keras')

predictions = model . predict (X_test_n)

score = model . evaluate ( X_test_n , y_test_categorical , verbose =0 )

y_test_p_binary = np.where(y_test_categorical > 0.5, 1, 0)
cm = confusion_matrix(y_test, y_test_p_binary)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

