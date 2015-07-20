from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D

'''
    Train a ConvNet on the 2D textured blocks data, using multivariate
    regression to produce a coordinate prediction.

    Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_regression_2d.py
'''

batch_size = 64
nb_classes = 10
nb_epoch = 1


def load_data():
    X_test_file = '/Users/cosmo/rendering/2d_multipixel_on_off_input_test.pkl'
    y_test_file_multivariate = '/Users/cosmo/rendering/2d_multipixel_on_off_labels_test_multivariate.pkl'

    textured_blocks_data = np.load(X_test_file)
    textured_blocks_multivariate_labels = np.load(y_test_file_multivariate)

    # Split into training and test sets
    m = textured_blocks_multivariate_labels.shape[0]
    num_test = round(0.20 * m)
    num_train = m - num_test
    indices = np.array(list(range(m)))
    np.random.shuffle(indices)

    # Reshape input from square images into flat vectors
    image_length = textured_blocks_data.shape[1] * textured_blocks_data.shape[2]
    textured_blocks_data = textured_blocks_data.reshape(m, image_length)
    label_length = textured_blocks_multivariate_labels.shape[1]

    X_train = np.zeros([num_train, image_length])
    y_train = np.zeros([num_train, label_length])
    X_test = np.zeros([num_test, image_length])
    y_test = np.zeros([num_test, label_length])

    for i in range(num_train):
        X_train[i] = textured_blocks_data[indices[i]]
        y_train[i] = textured_blocks_multivariate_labels[indices[i]]
    for i in range(num_test):
        X_test[i] = textured_blocks_data[indices[num_train + i]]
        y_test[i] = textured_blocks_multivariate_labels[indices[num_train + i]]

    return (X_train, y_train), (X_test, y_test)


# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = load_data()

X_train = X_train.reshape(X_train.shape[0], 1, 100, 100)
X_test = X_test.reshape(X_test.shape[0], 1, 100, 100)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

model = Sequential()

model.add(Convolution2D(32, 1, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
model.add(Convolution2D(32, 32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(80000, 128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(128, 2))
model.compile(loss='mean_absolute_error', optimizer='rmsprop')

model.load_weights('cnn_regression_2d_textured_blocks.hdf5')

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save_weights('cnn_regression_2d_textured_blocks.hdf5', overwrite=True)
