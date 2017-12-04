from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import pprint
pp = pprint.PrettyPrinter()
import features_rnn

# Consistent testing
np.random.seed(7)

print "Extracting features..."
X, Y = features_rnn.getFeatures(cached=True, limit="")
print "Done."

num, dim = X.shape
# Train, Validation, Test set split: 0.7 0.21 0.09
X_train, X_total_test, Y_train, Y_total_test = train_test_split(X, Y, test_size=0.3, random_state=3)
X_val, X_test, Y_val, Y_test = train_test_split(X_total_test, Y_total_test, test_size=0.3, random_state=3)
train_num = X_train.shape[0]

def RNN():
    model = Sequential()
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model

def evaluateModel(m, val=False, eta=500, batch=160):
    # Runs model given parameters
    m.fit(X_train.todense(), Y_train, epochs=eta, batch_size=batch)
    if val:
        results =  m.evaluate(X_val.todense(), Y_val, verbose=0)
    else:
        results =  m.evaluate(X_test.todense(), Y_test, verbose=0)
    return results
