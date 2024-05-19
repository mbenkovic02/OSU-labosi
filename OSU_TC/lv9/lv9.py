from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
import numpy as np
import tensorflow as tf

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

num_classes = 10

X_train_s = X_train.astype("float32") / 255
X_test_s = X_test.astype("float32") / 255

X_train_n = np.expand_dims(X_train_s, axis=-1)
X_test_n = np.expand_dims(X_test_s, axis=-1)

y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)

print(f"{X_train_n.shape} {X_test_n.shape}")

model = keras.Sequential()
model.add(layers.Input(shape=(32,32,3)))
model.add(layers.Conv2D(10, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(15, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", ])

my_callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience = 12, verbose = 1),
    keras.callbacks.TensorBoard(log_dir = 'logs/cnn', update_freq = 100)
]

model.fit(X_train_n, y_train_s, epochs = 50, batch_size = 64, callbacks = my_callbacks, validation_split = 0.1)