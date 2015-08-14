from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D

'''
    Train a ConvNet on the 3D Minecraft single block dataset, using
    multivariate regression to produce a 3D coordinate prediction
    given an input image.

    Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python convnet_regression_minecraft.py
'''

LOAD_SAVED_MODEL = False
SAVED_MODEL = 'cnn_regression_minecraft_3d_128x128_scaled_test.hdf5'
INPUT_IMAGES = '/Users/cosmo/mldata/regression_1/images.npy'
LABELS = '/Users/cosmo/mldata/regression_1/labels.npy'
IMAGE_WIDTH = 128

batch_size = 32
nb_epoch = 1


def load_data():
    X_train = np.load('train.npy')
    y_train = np.load('train_labels.npy')
    X_test  = np.load('test.npy')
    y_test  = np.load('test_labels.npy')

    return (X_train, y_train), (X_test, y_test)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = load_data()

X_train = X_train.reshape(X_train.shape[0], 1, IMAGE_WIDTH, IMAGE_WIDTH)
X_test = X_test.reshape(X_test.shape[0], 1, IMAGE_WIDTH, IMAGE_WIDTH)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

# Scale the input image data to be between -1 and 1
X_train /= 255
X_train -= 0.50
X_train *= 2
X_test /= 255
X_test -= 0.50
X_test *= 2

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Construct the model
model = Sequential()

# Convolutional layers
model.add(Convolution2D(32, 1, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
model.add(Convolution2D(32, 32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())

# Fully connected layer
# model.add(Dense(524288, 128))  # for 256x256
model.add(Dense(131072, 128))  # for 128x128
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Regression output layer
model.add(Dense(128, 3))

model.compile(loss='mean_absolute_error', optimizer='rmsprop')

if LOAD_SAVED_MODEL:
    model.load_weights(SAVED_MODEL)

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=1, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save_weights(SAVED_MODEL, overwrite=False)
