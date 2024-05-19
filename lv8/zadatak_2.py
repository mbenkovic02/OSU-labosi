import numpy as np
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from keras import models


model = models.load_model('zadatak_1_model.keras')

(x_train, y_train),(x_test, y_test) = keras.datasets.mnist.load_data()

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

predictions = model.predict(x_test_s)
prediction_labels = np.argmax(predictions, axis=1)

wrong_labels = np.where(prediction_labels != y_test)[0]

for i in range(16):
    index = wrong_labels[i]
    plt.subplot(4, 4, i + 1)
    plt.imshow(x_test[index].reshape(28, 28))
    plt.title(f'IS: {y_test[index]}, PREDICTED: {prediction_labels[index]}')
    plt.axis("off")

plt.show()


print('===============')
print(f"Wrong labels: {wrong_labels.shape[0]} / {prediction_labels.shape[0]}")
print(f"Accuracy: {(prediction_labels.shape[0] - wrong_labels.shape[0]) / prediction_labels.shape[0]}")
print('===============')