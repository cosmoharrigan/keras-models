from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337) # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

import matplotlib.pyplot as plt
import matplotlib.cm as cm


batch_size = 128
nb_classes = 10*10
# nb_epoch = 20
nb_epoch = 1000


def load_single_block_data():
    # output_file = '/Users/cosmo/blocks-models/minecraft_single_block_images_grayscale_128x128.pkl'  # X
    # X_file = '/Users/cosmo/blocks-models/output.pkl'  # X
    # input_file = '/Users/cosmo/blocks-models/input_block_type_only.pkl'  # y
    # y_file = '/Users/cosmo/nn/data/rough_position.pkl'  # y
    X_file = '/Users/cosmo/rendering/2d_multipixel_on_off_input.pkl'  # X
    y_file = '/Users/cosmo/rendering/2d_multipixel_on_off_labels.pkl'  # X

    X_test_file = '/Users/cosmo/rendering/2d_multipixel_on_off_input_test.pkl'
    y_test_file = '/Users/cosmo/rendering/2d_multipixel_on_off_labels_test.pkl'

    y_data = np.load(y_file)
    X_data = np.load(X_file)
    X_test_data = np.load(X_test_file)
    y_test_data = np.load(y_test_file)

    m = len(y_data)
    m_test = len(y_test_data)

    x_length = 100*100

    X_train = X_data.reshape(m, x_length)
    y_train = y_data
    X_test = X_test_data.reshape(m_test, x_length)
    y_test = y_test_data

    return (X_train, y_train), (X_test, y_test)


# the data, shuffled and split between tran and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

(X_train, y_train), (X_test, y_test) = load_single_block_data()

# X_train = X_train.reshape(8000, 10)
# X_test = X_test.reshape(2000, 10)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
# X_train /= 255
# X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

print('x shape ', X_train.shape)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# import IPython
# IPython.embed()

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
    # import IPython
    # IPython.embed()
    if prediction.argmax() == Y_test[idx].argmax():
        confidences[idx] = prediction[:, prediction.argmax()][0]
    else:
        # print('ERROR on sample ', idx)
        pass

print('mean confidence: ', np.mean(confidences))
# plt.hist(confidences)
# plt.show()

model.save_weights('mlp_experiment_improved2.hdf5', overwrite=True)

"""
100 epochs, 6.8%, 3.9 cost (more hidden units?)
1000 epochs, 19.0%, 3.0 cost
1100 epochs, 19.2%, 3.0 cost, 2.7% mean confidence
1600 epochs, 20.3%, 2.9 cost, 3.0% mean confidence
----- made it a 1-hidden-layer network with 2 units in the hidden layer

Epoch 499
0s - loss: 3.7014 - acc: 0.0600 - val_loss: 3.6334 - val_acc: 0.0933
Test score: 3.63343827128
Test accuracy: 0.0933
Evaluating learned model:
10000/10000 [==============================] - 1s
[3.6334382712764643, 0.093299999999999994]
Evaluating one test sample at a time:
mean confidence:  0.00636033116215

Epoch 499
0s - loss: 3.5446 - acc: 0.0700 - val_loss: 3.4259 - val_acc: 0.1215
Test score: 3.42592637076
Test accuracy: 0.1215
x test max:  1.0
Evaluating learned model:
10000/10000 [==============================] - 1s
[3.4259263707649459, 0.1215]
Evaluating one test sample at a time:
mean confidence:  0.010493721105

Epoch 499
0s - loss: 3.4663 - acc: 0.1100 - val_loss: 3.3149 - val_acc: 0.1693
Test score: 3.31488870725
Test accuracy: 0.1693
x test max:  1.0
Evaluating learned model:
10000/10000 [==============================] - 1s
[3.3148887072529796, 0.16930000000000001]
Evaluating one test sample at a time:
mean confidence:  0.0174075847716

Epoch 999
0s - loss: 3.5072 - acc: 0.1000 - val_loss: 3.2147 - val_acc: 0.1482
Test score: 3.2147267954
Test accuracy: 0.1482
x test max:  1.0
Evaluating learned model:
10000/10000 [==============================] - 1s
[3.214726795398807, 0.1482]
Evaluating one test sample at a time:
mean confidence:  0.0188703956692
"""
