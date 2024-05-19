"""Zadatak 5.5.1 Skripta zadatak_1.py generira umjetni binarni klasifikacijski problem s dvije
ulazne velicine. Podaci su podijeljeni na skup za u ˇ cenje i skup za testiranje modela. ˇ
a) Prikažite podatke za ucenje u ˇ x1 −x2 ravnini matplotlib biblioteke pri cemu podatke obojite ˇ
s obzirom na klasu. Prikažite i podatke iz skupa za testiranje, ali za njih koristite drugi
marker (npr. ’x’). Koristite funkciju scatter koja osim podataka prima i parametre c i
cmap kojima je moguce de ´ finirati boju svake klase.
b) Izgradite model logisticke regresije pomo ˇ cu scikit-learn biblioteke na temelju skupa poda- ´
taka za ucenje. ˇ
c) Pronadite u atributima izgra ¯ denog modela parametre modela. Prikažite granicu odluke ¯
naucenog modela u ravnini ˇ x1 − x2 zajedno s podacima za ucenje. Napomena: granica ˇ
odluke u ravnini x1 −x2 definirana je kao krivulja: θ0 +θ1x1 +θ2x2 = 0.
d) Provedite klasifikaciju skupa podataka za testiranje pomocu izgra ´ denog modela logisti ¯ cke ˇ
regresije. Izracunajte i prikažite matricu zabune na testnim podacima. Izra ˇ cunate to ˇ cnost, ˇ
preciznost i odziv na skupu podataka za testiranje.
e) Prikažite skup za testiranje u ravnini x1 −x2. Zelenom bojom oznacite dobro klasi ˇ ficirane
primjere dok pogrešno klasificirane primjere oznacite crnom bojom."""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


#a)

plt.scatter(X_train[:,0], X_train[:,1], c="red")
plt.scatter(X_test[:,0], X_test[:,1], c="blue", marker='*')
plt.show()

#b)

LogRegression_model = LogisticRegression ()
LogRegression_model . fit ( X_train , y_train )
# predikcija na skupu podataka za testiranje
y_test_p = LogRegression_model . predict ( X_test )


#c)

# Dobivanje parametara modela
b = LogRegression_model.intercept_[0]
w1,w2 = LogRegression_model.coef_.T

#b = LogRegression_model.intercept_[0]: Dobiva vrijednost pristranosti (intercept) 
#θ 0 iz logističkog regresijskog modela.
#w1, w2 = LogRegression_model.coef_.T: Dobiva koeficijente 
#θ 1 i θ 2
#transponirala matrica koeficijenata jer su dobiveni kao vektor 
#redaka, a želimo ih kao vektor stupaca.

c = -b/w2
m = -w1/w2

#c = -b / w2: Izračunava y-koordinatu (x2) početka granice odluke na 
#osnovi pristranosti i koeficijenta za x2.
#m = -w1 / w2: Izračunava nagib granice odluke na osnovi koeficijenata 
#θ 1 i θ 2.

xmin, xmax = -4, 4
ymin, ymax = -4, 4
xd = np.array([xmin, xmax])
yd = m*xd+c

plt.plot(xd,yd, 'k', lw=1, ls='--')
plt.fill_between(xd, yd, ymin,color='orange', alpha=0.2)
plt.fill_between(xd, yd, ymax, color='blue', alpha=0.2)
plt.show()

#xd = np.array([xmin, xmax]): Definira raspon x-koordinata za iscrtavanje granice odluke.
#yd = m * xd + c: Računa odgovarajuće y-koordinate (x2) za iscrtavanje granice odluke na temelju dobivenih koeficijenata i pristranosti.
#plt.plot(xd, yd, 'k', lw=1, ls='--'): Iscrtava granicu odluke kao crnu prekinutu liniju.
#plt.fill_between(xd, yd, ymin, color='orange', alpha=0.2): Boji područje ispod granice odluke narančastom bojom.
#plt.fill_between(xd, yd, ymax, color='blue', alpha=0.2): Boji područje iznad granice odluke plavom bojom.


#d)

cm = confusion_matrix ( y_test , y_test_p )
print (" Matrica zabune : " , cm )
disp = ConfusionMatrixDisplay ( confusion_matrix ( y_test , y_test_p ) )
disp . plot ()
plt . show ()


#e)

y1 = (y_test==y_test_p)
y0 = (y_test!=y_test_p)

X_false = []

for i in range(len(y_test)):
    if y_test[i] != y_test_p[i]:
        X_false.append([X_test[i, 0], X_test[i, 1]])

X_false = np.array(X_false)
print(X_false)

plt.scatter(X_test[:,0], X_test[:, 1])
plt.scatter(X_false[:,0], X_false[:,1], color='green')
plt.show()

#y1 = (y_test == y_test_p): Stvara niz boolean vrijednosti gdje je True ako su stvarne vrijednosti u testnom skupu podataka jednake predviđenim vrijednostima, a False inače.
#y0 = (y_test != y_test_p): Stvara niz boolean vrijednosti gdje je True ako su stvarne vrijednosti u testnom skupu podataka različite od predviđenih vrijednosti, a False inače.
#X_false = []: Inicijalizira praznu listu za spremanje pogrešno klasificiranih podataka.
#for i in range(len(y_test)):: Iterira kroz indekse testnih podataka.
#if y_test[i] != y_test_p[i]:: Provjerava je li stvarna vrijednost podataka na trenutnom indeksu različita od predviđene vrijednosti.
#X_false.append([X_test[i, 0], X_test[i, 1]]): Ako je došlo do pogrešne klasifikacije, dodaje značajke tog podatka (x1 i x2) u listu X_false.
#X_false = np.array(X_false): Pretvara listu pogrešno klasificiranih podataka u NumPy array.
#print(X_false): Ispisuje pogrešno klasificirane podatke.
#plt.scatter(X_test[:,0], X_test[:, 1]): Iscrtava sve testne podatke na grafu.
#plt.scatter(X_false[:,0], X_false[:,1], color='green'): Iscrtava pogrešno klasificirane podatke na grafu zelenom bojom.
#plt.show(): Prikazuje graf.