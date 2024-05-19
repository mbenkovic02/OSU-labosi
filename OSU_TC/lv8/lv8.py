from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split
import os

model = keras.Sequential()
model.add(layers.Input(shape=(2, )))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.summary()

X = np.random.randint(0, 10, size=(100, 2))
y = np.random.randint(0, 10, size=100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", ])
batch_size = 32
epochs = 20
history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, validation_split = 0.1)

predictions = model.predict(X_test)

score = model.evaluate(X_test, y_test, verbose = 0)

model_dir = os.path.join(os.getcwd(),"FCN")
os.makedirs(model_dir, exist_ok=True)
model_dir = os.path.join(model_dir, "model.keras")
model.save(model_dir)
del model

model = load_model(model_dir)
model.summary()
