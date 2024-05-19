"""Zadatak 2.4.1 Pomocu funkcija ´ numpy.array i 
matplotlib.pyplot pokušajte nacrtati sliku
2.3 u okviru skripte zadatak_1.py. Igrajte se sa slikom, promijenite boju linija, debljinu linije i
sl."""

import numpy as np
import matplotlib . pyplot as plt

x= np.array([1,3,3,2,1]) # ili square = [(1, 1), (3, 1), (3, 2), (2, 2), (1, 1)]
y= np.array([1,1,2,2,1]) #     x, y = zip(*square)

plt.plot(x, y, 'r', marker=".", markersize=6)


plt . axis ([0 ,4 ,0 , 4])
plt . xlabel ("x os")
plt . ylabel ("y os")
plt . title ( "Zadatak 1")
plt . show ()