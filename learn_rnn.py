from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pprint
pp = pprint.PrettyPrinter()
import features_rnn
import util

# Consistent testing
np.random.seed(7)
tf.set_random_seed(7)

print "Extracting features..."
X, Y = features_rnn.getFeatures(cached=True, limit="")
print "Done."

# Train, Validation, Test set split: 0.7 0.21 0.09
# X are numpy arrays of sparse matrices
X_train, X_total_test, Y_train, Y_total_test = train_test_split(X, Y, test_size=0.3, random_state=3)
X_val, X_test, Y_val, Y_test = train_test_split(X_total_test, Y_total_test, test_size=0.3, random_state=3)
num = X_train.shape[0] # 2700
seq_len, dim = X_train[0].shape # 200, 22

def RNN():
    model = Sequential()
    model.add(LSTM(22, input_shape=(seq_len, dim)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model


def ensemble():
    rnn = Sequential()
    rnn.add(LSTM(22, input_shape=(seq_len, dim)))
    rnn.add(Dense(1, activation='sigmoid'))

    ann = Sequential()
    ann.add(Dense(hidden_nodes, input_dim=dim, kernel_initializer='normal', activation='relu'))
    ann.add(Dense(1, activation='sigmoid'))

    model = Sequential()
    model.add(Merge([rnn, ann], mode='concat'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

def evaluateModel(m, val=False, eta=500, batch=160):
    # Runs model given parameters
    m.fit(X_train, Y_train, epochs=eta, batch_size=batch)
    if val:
        results =  m.evaluate(X_val, Y_val, verbose=1)
    else:
        results =  m.evaluate(X_test, Y_test, verbose=1)
    return results

rnn = RNN()
res = evaluateModel(rnn, False, 500, 500)
pp.pprint(res)
