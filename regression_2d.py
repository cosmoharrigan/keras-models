# Based on: https://github.com/fchollet/keras/issues/108#issuecomment-100585999

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation


def load_data():
    X_file = '/Users/cosmo/rendering/2d_multipixel_on_off_input.pkl'  # X
    y_file = '/Users/cosmo/rendering/2d_multipixel_on_off_labels.pkl'  # X

    X_test_file = '/Users/cosmo/rendering/2d_multipixel_on_off_input_test.pkl'
    y_test_file = '/Users/cosmo/rendering/2d_multipixel_on_off_labels_test.pkl'

    y_file_multivariate = '/Users/cosmo/rendering/2d_multipixel_on_off_labels_multivariate.pkl'
    y_test_file_multivariate = '/Users/cosmo/rendering/2d_multipixel_on_off_labels_test_multivariate.pkl'

    y_data = np.load(y_file)
    X_data = np.load(X_file)
    X_test_data = np.load(X_test_file)
    y_test_data = np.load(y_test_file)
    t_data_multivariate = np.load(y_file_multivariate)
    y_test_data_multivariate = np.load(y_test_file_multivariate)

    m = len(y_data)
    m_test = len(y_test_data)

    x_length = 100*100

    # X_train = X_data.reshape(m, x_length)
    # y_train = y_data
    # X_test = X_test_data.reshape(m_test, x_length)
    # y_test = y_test_data

    X_train = X_test_data.reshape(m_test, x_length)
    X_test = X_test_data.reshape(m_test, x_length)
    y_train = y_test_data_multivariate
    y_test = y_test_data_multivariate

    return (X_train, y_train), (X_test, y_test)

(X_train, y_train), (X_test, y_test) = load_data()

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

# Normalize
# X_train /= 10
# X_test /= 10
# y_train /= 10
# y_test /= 10

print('X_train', X_train.shape)
print('y_train', y_train.shape)
print('X_test', X_test.shape)
print('y_test', y_test.shape)

model = Sequential()
# model.add(Dense(10, 64))
model.add(Dense(100*100, 2))
model.add(Activation('tanh'))
model.add(Dense(2, 2))
model.compile(loss='mean_absolute_error', optimizer='rmsprop')

model.load_weights('regression_2d_constrained_unscaled.hdf5')

model.fit(X_train, y_train, nb_epoch=200, batch_size=16)
score = model.evaluate(X_test, y_test, batch_size=16)

print(score)

model.save_weights('regression_2d_constrained_unscaled.hdf5', overwrite=True)

m_test = 10000
prediction = model.predict(X_test)
print('prediction shape ', prediction.shape)

num_incorrect = 0
for i in range(m_test):
    estimate = np.around(prediction[i])
    correct_answer = y_test[i, :]

    if not np.array_equal(estimate, correct_answer):
        num_incorrect += 1

print('Success rate: {0} correct out of {1} ({2})'.format(m_test - num_incorrect, m_test, (m_test - num_incorrect) / m_test))
