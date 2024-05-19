"""Zadatak 2.4.2 Datoteka data.csv sadrži mjerenja visine i mase provedena na muškarcima i
ženama. Skripta zadatak_2.py ucitava dane podatke u obliku numpy polja ˇ data pri cemu je u ˇ
prvom stupcu polja oznaka spola (1 muško, 0 žensko), drugi stupac polja je visina u cm, a treci´
stupac polja je masa u kg.
18 Poglavlje 2. Rad s bibliotekama Numpy i Matplotlib
a) Na temelju velicine numpy polja data, na koliko osoba su izvršena mjerenja? ˇ
b) Prikažite odnos visine i mase osobe pomocu naredbe ´ matplotlib.pyplot.scatter.
c) Ponovite prethodni zadatak, ali prikažite mjerenja za svaku pedesetu osobu na slici.
d) Izracunajte i ispišite u terminal minimalnu, maksimalnu i srednju vrijednost visine u ovom ˇ
podatkovnom skupu.
e) Ponovite zadatak pod d), ali samo za muškarce, odnosno žene. Npr. kako biste izdvojili
muškarce, stvorite polje koje zadrži bool vrijednosti i njega koristite kao indeks retka.
ind = (data[:,0] == 1)"""

import numpy as np
import matplotlib . pyplot as plt

data = np.loadtxt("lv2\data.csv", delimiter=",", dtype="str")

data = data[1::]

data = np.array(data, np.float64)

#a)

print(f"Broj mjerenja:", len(data))


#b)

height, weight = data[:, 1], data[:, 2]

plt.scatter(height, weight)
plt.show()


min_height = np.min(height)
max_height = np.max(height)
avg_height = np.mean(height)

#c)

height, weight= data[0::50, 1], data[0::50, 2]

plt.scatter(height, weight)
plt.show()

#d)

print("Min height:", min_height)
print("Max height:", max_height)
print("Average height:", avg_height)


males = data[data[:, 0] == 1]
females = data[data[:, 0] == 0]

print(males)

print(f"Males:\n Max:{males[:, 1].max()}\nMin:{males[:, 1].min()}\nAverage:{males[:, 1].mean()}")
print(f"Females:\n Max:{females[:, 1].max()}\nMin:{females[:, 1].min()}\nAverage:{females[:, 1].mean()}")