"""Zadatak 2.4.4 Napišite program koji ce kreirati sliku koja sadrži ´ cetiri kvadrata crne odnosno ˇ
bijele boje (vidi primjer slike 2.4 ispod). Za kreiranje ove funkcije koristite numpy funkcije
zeros i ones kako biste kreirali crna i bijela polja dimenzija 50x50 piksela. Kako biste ih složili
u odgovarajuci oblik koristite numpy funkcije ´ hstack i vstack.
"""

import numpy as np
import matplotlib.pyplot as plt

zero = np.zeros((50,50))
one = np.ones((50,50))

top= np.hstack((one, zero))
bottom = np.hstack((zero,one))
matrix= np.vstack((top, bottom))

plt.imshow(matrix, cmap="gray")
plt.show()