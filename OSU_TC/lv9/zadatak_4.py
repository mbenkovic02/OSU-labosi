import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt


def do_learning(name, batch_size = 64, optimizer = 'adam', use_small_model = False, use_small_dataset = False):
    print(name)

    # ucitaj CIFAR-10 podatkovni skup
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # prikazi 9 slika iz skupa za ucenje
    plt.figure()
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.xticks([]),plt.yticks([])
        plt.imshow(X_train[i])

    plt.show()

    if use_small_dataset:
        length = int(X_train.shape[0] / 2)
        X_train = X_train[0:length,:]
        y_train = y_train[0:length,:]
        

    # pripremi podatke (skaliraj ih na raspon [0,1]])
    X_train_n = X_train.astype('float32')/ 255.0
    X_test_n = X_test.astype('float32')/ 255.0

    # 1-od-K kodiranje
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # CNN mreza
    model = keras.Sequential()

    model.add(layers.Input(shape=(32,32,3)))
    if use_small_model:
        model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(10, activation='softmax'))
    else:
        model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.3))
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.3))
        model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(500, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(10, activation='softmax'))

    model.summary()

    # definiraj listu s funkcijama povratnog poziva
    my_callbacks = [
        keras.callbacks.TensorBoard(log_dir = f'logs/cnn_{name}', update_freq = 100),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    ]

    model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    model.fit(X_train_n,
                y_train,
                epochs = 40,
                batch_size = batch_size,
                callbacks = my_callbacks,
                validation_split = 0.1)


    score = model.evaluate(X_test_n, y_test, verbose=0)
    print(f'Tocnost na testnom skupu podataka: {100.0*score[1]:.2f}')

do_learning("base_line")
do_learning("small_batch", batch_size=32)
# do_learning("different_learning_rate", optimizer='adamax')
# do_learning("less_layers", use_small_model=True)
# do_learning("small_dataset", use_small_dataset=True)

'''
BASELINE:
    Prestao nakon 16 epoha - najbolji model nakon 11te epohe
    Tocnost na testu: 75.86
    Vrijeme po epohi: 11 - 12 s

SMALL BATCH:
    Prestao nakon 18 epoha - najbolji model nakon 13te epohe
    Tocnost na testnom skupu podataka 76.49
    Vrijeme po epohi: 15 - 16 s

DIFFERENT LEARNING RATE:
    koristen ADAMAX optimizer kako bi se dobio drugaciji learning rate
    Prestao nakon 30 epoha, najtocniji na 25toj
    Tocnost na testnom skupu 77.75
    Vrijeme po epohi: 11 - 12 s

LESS LAYERS:
    Prestao nakon 17 epoha, najtocniji na 12toj
    Tocnost na testnom skupu 71.31
    Vrijeme po epohi: 7-10 s

LESS DATA:
    Prestao nakon 17 epoha, najtocniji na 12toj
    Tocnost na testnom skupu 70.86
    Vrijeme po epohi 5-7 s
'''
