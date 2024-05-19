"""Zadatak 2.4.3 Skripta zadatak_3.py ucitava sliku ’ ˇ road.jpg’. Manipulacijom odgovarajuce´
numpy matrice pokušajte:
a) posvijetliti sliku,
b) prikazati samo drugu cetvrtinu slike po širini, ˇ
c) zarotirati sliku za 90 stupnjeva u smjeru kazaljke na satu,
d) zrcaliti sliku."""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt

img=plt.imread("road.jpg")

img = img[:,:,0].copy()

plt.figure()
plt.imshow(img, cmap="gray")
plt.show()

#a)

image1= np.add(img, 50, dtype=np.uint8)
plt.imshow(image1, cmap="gray")
plt.show()

#b)

width = len(img[0])

quarter_width=width//4

plt.imshow(img[ : , 1*quarter_width : 2*quarter_width], cmap="gray")
plt.show()

#c)

image2= np.rot90(img, 3)
plt.imshow(image2, cmap="gray")
plt.show()

#d)

image3= np.flip(img)
plt.imshow(image3, cmap="gray")
plt.show()