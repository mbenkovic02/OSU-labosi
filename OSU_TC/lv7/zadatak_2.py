import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
for i in range(1, 7):
    if i == 4:
        continue  # jer slike test_4 se ne učitava dobro valjda krivi format

    img = Image.imread(f"lv7\\imgs\\test_{i}.jpg")

    # prikazi originalnu sliku
    plt.figure()
    plt.title(f"Originalna slika test_{i}.jpg")
    plt.imshow(img)
    plt.tight_layout()
    plt.show()

    # pretvori vrijednosti elemenata slike u raspon 0 do 1
    img = img.astype(np.float64) / 255

    # transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
    w, h, d = img.shape
    img_array = np.reshape(img, (w * h, d))

    # rezultatna slika
    img_array_aprox = img_array.copy()

    km = KMeans(n_clusters=5, init="k-means++", n_init=5, random_state=0)
    km.fit(img_array_aprox)
    labels = km.predict(img_array_aprox)

    centroids = km.cluster_centers_

    img_array_aprox[:, 0] = centroids[labels][:, 0]
    img_array_aprox[:, 1] = centroids[labels][:, 1]
    img_array_aprox[:, 2] = centroids[labels][:, 2]
    img_array_aprox = np.reshape(img_array_aprox, (w, h, d))

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img)
    axarr[1].imshow(img_array_aprox)
    plt.tight_layout()
    plt.show()
