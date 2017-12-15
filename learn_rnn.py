from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Merge
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pprint
pp = pprint.PrettyPrinter()
import features.rnn
import features.nn
import util

# Consistent testing
np.random.seed(7)
tf.set_random_seed(7)

print "Extracting features..."
# Temporal Features
titles_train, X_train, Y_train, titles_test, X_test, Y_test, X_all, Y_all = features.rnn.getFeatures(cached=True, limit="")
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=3)
print "Done."

num = X_train.shape[0]
seq_len, dim = X_train[0].shape
print num, seq_len, dim

# Train and Test set are randomly distributed: 0.6, 0.24, 0.16
X_rand_train, X_rand_test, Y_rand_train, Y_rand_test = train_test_split(X_all, Y_all, test_size=0.4, random_state=3)
X_rand_val, X_rand_test, Y_rand_val, Y_rand_test = train_test_split(X_test, Y_test, test_size=0.4, random_state=3)

# Total Song Features
_, X_tot_train, Y_tot_train, _, X_tot_test, Y_tot_test, X_tot_all, Y_tot_all = features.nn.getFeatures(cached=True, limit="")
X_tot_train, X_tot_val, Y_tot_train, Y_tot_val = train_test_split(X_tot_train, Y_tot_train, test_size=0.3, random_state=3)

# Total Song Random
X_tot_rand_train, X_tot_rand_test, Y_tot_rand_train, Y_tot_rand_test = train_test_split(X_all, Y_all, test_size=0.4, random_state=3)
X_tot_rand_val, X_tot_rand_test, Y_tot_rand_val, Y_tot_rand_test = train_test_split(X_test, Y_test, test_size=0.4, random_state=3)

def RNN():
    model = Sequential()
    model.add(LSTM(18, input_shape=(seq_len, dim)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model

def ensemble():
    rnn = Sequential()
    rnn.add(LSTM(18, input_shape=(seq_len, dim)))
    rnn.add(Dense(1, activation='sigmoid'))

    ann = Sequential()
    ann.add(Dense(7, input_dim=dim, kernel_initializer='normal', activation='relu'))
    ann.add(Dense(1, activation='sigmoid'))

    model = Sequential()
    model.add(Merge([rnn, ann], mode='concat'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model

def evaluateModel(m, val=False, eta=500, batch=160):
    # Runs model given parameters
    m.fit(X_train, Y_train, epochs=eta, batch_size=batch)
    if val:
        results =  m.evaluate(X_val, Y_val, verbose=1)
    else:
        results =  m.evaluate(X_test, Y_test, verbose=1)
    print "TEMPORAL", results

    m.fit(X_rand_train, Y_rand_train, epochs=eta, batch_size=batch)
    if val:
        results_rand =  m.evaluate(X_rand_val, Y_rand_val, verbose=1)
    else:
        results_rand =  m.evaluate(X_rand_test, Y_rand_test, verbose=1)
    print "RANDOM", results_rand
    
    return results, results_rand

def evaluateEnsemble(ens):
    ens.fit([X_train, X_tot_train], Y_train, epochs=500, batch_size=500)
    results = ens.evaluate([X_test, X_tot_test], Y_test)
    print "TEMPORAL", results

    ens.fit([X_rand_train, X_tot_rand_train], Y_rand_train, epochs=500, batch_size=500)
    rand_results = ens.evaluate([X_rand_test, X_tot_rand_test], Y_rand_test)

    print "TEMPORAL", results
    print "RANDOM", rand_results

    return results, results_rand

print "ENSEMBLE"
ens = ensemble()
res = evaluateEnsemble(ens)
print "ENSEMBLE"
print "TEMPORAL", res[0]
print "RANDOM", res[1]
