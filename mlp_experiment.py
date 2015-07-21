from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

batch_size = 128
nb_classes = 10*10
# nb_epoch = 20
nb_epoch = 1000


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

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(100*100, 2))
model.add(Activation('relu'))

# ----
# model.add(Dropout(0.2))
# model.add(Dense(128, 128))
# model.add(Activation('relu'))

# model.add(Dropout(0.2))
# model.add(Dense(128, 2))
# model.add(Activation('relu'))
# ----

model.add(Dropout(0.2))
model.add(Dense(2, 100))
model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)

print(X_train.shape)
print(X_test.shape)

model.load_weights('mlp_experiment_improved2.hdf5')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

print('Evaluating learned model:')
result = model.evaluate(X_test, Y_test, batch_size=len(Y_test), show_accuracy=True, verbose=1)
print(result)

m_test = len(Y_test)
confidences = np.zeros(m_test)
print('Evaluating one test sample at a time:')
for idx in range(m_test):
    prediction = model.predict(X_test[idx].reshape(1, 10000))
    if prediction.argmax() == Y_test[idx].argmax():
        confidences[idx] = prediction[:, prediction.argmax()][0]

print('mean confidence: ', np.mean(confidences))

model.save_weights('mlp_experiment_improved2.hdf5', overwrite=True)
