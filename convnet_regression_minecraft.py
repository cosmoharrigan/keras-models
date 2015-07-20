from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337) # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D

'''
    Train a ConvNet on the 3D Minecraft single block dataset, using
    multivariate regression to produce a 3D coordinate prediction
    given an input image.

    Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_regression_minecraft.py
'''

SAVED_MODEL = 'cnn_regression_minecraft_3d_128x128_scaled.hdf5'
INPUT_IMAGES = '/Users/cosmo/blocks-models/minecraft_single_block_images_grayscale_128x128.pkl'
LABELS = '/Users/cosmo/blocks-models/input_position_only.pkl'
IMAGE_WIDTH = 128

batch_size = 32
nb_epoch = 1


def load_data():
    X_file = INPUT_IMAGES
    y_file = LABELS

    X_data = np.load(X_file)
    y_data = np.load(y_file)

    # Split into training and test sets
    m = X_data.shape[0]
    num_test = round(0.10 * m)
    num_train = m - num_test
    indices = np.array(list(range(m)))
    np.random.shuffle(indices)

    # Optional: discard Z coordinate information to make it a 2-D regression problem
    # y_data = y_data[:, 0:2]

    image_length = X_data.shape[1]
    label_length = y_data.shape[1]

    X_train = np.zeros([num_train, image_length])
    y_train = np.zeros([num_train, label_length])
    X_test = np.zeros([num_test, image_length])
    y_test = np.zeros([num_test, label_length])

    for i in range(num_train):
        X_train[i] = X_data[indices[i]]
        y_train[i] = y_data[indices[i]]
    for i in range(num_test):
        X_test[i] = X_data[indices[num_train + i]]
        y_test[i] = y_data[indices[num_train + i]]

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

model.load_weights(SAVED_MODEL)

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=1, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save_weights(SAVED_MODEL, overwrite=True)
