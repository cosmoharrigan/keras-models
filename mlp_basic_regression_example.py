# Based on: https://github.com/fchollet/keras/issues/108#issuecomment-100585999

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation


def load_data():
    data_folder = '/Users/cosmo/nn/data/'
    data_file = 'ex1data1.csv'
    data = np.genfromtxt(data_folder + data_file, delimiter=',', dtype=np.float64)
    X = data[:, 0]
    y = data[:, 1]
    X_train = X
    y_train = y
    X_test = X
    y_test = y
    return (X_train.reshape(97, 1), y_train), (X_test.reshape(97, 1), y_test)

(X_train, y_train), (X_test, y_test) = load_data()

print(X_train.shape)

model = Sequential()
# model.add(Dense(10, 64))
model.add(Dense(1, 64))
model.add(Activation('tanh'))
model.add(Dense(64, 1))
model.compile(loss='mean_absolute_error', optimizer='rmsprop')

model.fit(X_train, y_train, nb_epoch=10000, batch_size=16)
score = model.evaluate(X_test, y_test, batch_size=16)

print(score)

prediction = model.predict(np.asarray([16]).reshape(1, 1))
print(prediction)

# cost after 10000: 2.19
# cost after 1000:  2.20
# predictions for x=16 (should be around 15): 100->10, 1000->13, 10000->16.3


