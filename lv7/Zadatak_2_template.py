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



##########




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
