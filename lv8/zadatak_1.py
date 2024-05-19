import numpy as np
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# TODO: prikazi nekoliko slika iz train skupa

num_of_samples = 5
plt.figure()
for i in range(num_of_samples):
    plt.subplot(1, num_of_samples, i+1)
    plt.imshow(x_train[i])
plt.show()



# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)


# TODO: kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu

model = keras . Sequential ()
model . add ( layers . Input ( input_shape ))
model.add(layers.Flatten())       #za error:  Arguments `target` and `output` must have the same rank (ndim). Received: target.shape=(None, 10), output.shape=(None, 28, 28, 10)
model . add ( layers . Dense (100 , activation ="relu") )
model . add ( layers . Dense (50 , activation ="sigmoid") )
model . add ( layers . Dense (10 , activation ="softmax") )
model . summary ()

# TODO: definiraj karakteristike procesa ucenja pomocu .compile()

model . compile ( loss ="categorical_crossentropy" , optimizer ="adam", metrics = ["accuracy", ])


# TODO: provedi ucenje mreze

batch_size = 32
epochs = 20
history = model . fit ( x_train_s , y_train_s , batch_size = batch_size , epochs = epochs , validation_split = 0.1)
predictions = model . predict (x_test_s)



# TODO: Prikazi test accuracy i matricu zabune

score = model . evaluate ( x_test_s , y_test_s , verbose =0 )

disp = ConfusionMatrixDisplay(confusion_matrix(np.argmax(y_test_s, axis=1), np.argmax(predictions, axis=1)))
disp.plot()
plt.show()
# TODO: spremi model

model.save('zadatak_1_model.keras')