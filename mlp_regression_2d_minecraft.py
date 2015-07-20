# Based on: https://github.com/fchollet/keras/issues/108#issuecomment-100585999

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation


def load_data():
    X_file = '/Users/cosmo/blocks-models/minecraft_single_block_images_grayscale_128x128.pkl'
    y_file = '/Users/cosmo/blocks-models/input_position_only.pkl'

    X_data = np.load(X_file)
    y_data = np.load(y_file)

    m = len(y_data)

    x_length = 128*128

    # # discard Z coordinate information for now
    y_data = y_data[:, 0:2]

    X_train = X_data.reshape(m, x_length)
    y_train = y_data

    X_test = X_train
    y_test = y_train

    return (X_train, y_train), (X_test, y_test)

(X_train, y_train), (X_test, y_test) = load_data()

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

print(X_test[0].max())

# Normalize
X_train /= 255
X_train -= 0.5
X_test /= 255
X_test -= 0.5
y_train /= 10
y_train -= 0.5
y_test /= 10
y_test -= 0.5

print('X_train', X_train.shape)
print('y_train', y_train.shape)
print('X_test', X_test.shape)
print('y_test', y_test.shape)

model = Sequential()
# model.add(Dense(10, 64))
model.add(Dense(128*128, 100))
# model.add(Activation('tanh'))
model.add(Activation('relu'))
model.add(Dense(100, 2))
model.compile(loss='mean_absolute_error', optimizer='rmsprop')

model.load_weights('regression_2d_minecraft.hdf5')

model.fit(X_train, y_train, nb_epoch=500, batch_size=16)
score = model.evaluate(X_test, y_test, batch_size=16)

print(score)

model.save_weights('regression_2d_minecraft.hdf5', overwrite=True)

m_test = 8080
prediction = model.predict(X_test)
print('prediction shape ', prediction.shape)

num_incorrect = 0
for i in range(m_test):
    estimate = np.around(prediction[i])
    correct_answer = y_test[i, :]

    if not np.array_equal(estimate, correct_answer):
        num_incorrect += 1

print('Success rate: {0} correct out of {1} ({2})'.format(m_test - num_incorrect, m_test, (m_test - num_incorrect) / m_test))
print(prediction[i])
print(y_test[i])
