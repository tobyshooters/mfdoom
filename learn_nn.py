import tensorflow as tf
import numpy as np
# Consistent testing
np.random.seed(7)
tf.set_random_seed(7)
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pprint
pp = pprint.PrettyPrinter()
import features_nn

print "Extracting features..."
titles, X, Y = features_nn.getFeatures(cached=True, limit="")
print "Done."

num, dim = X.shape
# Train, Validation, Test set split: 0.7 0.21 0.09
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=3)
X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.3, random_state=3)
train_num = X_train.shape[0]

def linearRegression(theta=0.01):
    model = Sequential()
    model.add(Dense(1, input_dim=dim, kernel_initializer='normal', activation='sigmoid',
        kernel_regularizer=regularizers.l1(theta)))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model

def neuralNetwork(hidden_nodes=5):
    model = Sequential()
    model.add(Dense(hidden_nodes, input_dim=dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model

def deepNeuralNetwork():
    model = Sequential()
    model.add(Dense(9, input_dim=dim, kernel_initializer='normal'))
    model.add(Dense(5, kernel_initializer='normal'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model

def evaluateModel(m, val=False, eta=500, batch=160):
    # Runs model given parameters
    m.fit(X_train.todense(), Y_train, epochs=eta, batch_size=batch, 
            verbose=1, shuffle=False)
    if val:
        results =  m.evaluate(X_val.todense(), Y_val, verbose=0)
    else:
        results =  m.evaluate(X_test.todense(), Y_test, verbose=0)

    prediction = m.predict(X_test.todense())
    for i, predict in enumerate(prediction):
        print "--------------------------------"
        print "Title:      ", titles[i]
        print "Prediction: ", predict
        print "Actual:     ", Y_test[i]
        print "--------------------------------"

    # pp.pprint(m.summary())
    # for layer in m.layers:
    #     pp.pprint(layer.get_config())
    #     pp.pprint(layer.get_weights()[0])

    return results

def resultAnalysis(result):
    # Given a dictionary of (params) => (MSE, MAE), get best of each
    best_mse = sorted(result.iteritems(), key=lambda (k, v): v[0])[0]
    print "Best MSE: ", best_mse # (500, 160)
    best_mae = sorted(result.iteritems(), key=lambda (k, v): v[1])[0]
    print "Best MAE: ", best_mae # (700, 110)
    best = sorted(result.iteritems(), key=lambda (k, v): v[0] + v[1])[0]
    print "Best Overall: ", best

def hyperOptimize():
    # Get best eta and batch_size for linear regression using validation set
    ln = linearRegression(0)
    result = {}
    for eta in range(600, 601, 100):
        for batch in range(400, 1011, 50):
            res = evaluateModel(ln, True, eta, batch)
            result[(eta, batch)] = res
            print eta, batch, res
    resultAnalysis(result)

def optimizeRegularization():
    result = {}
    for coef in range(4):
        theta = 10**(-coef)
        ln = linearRegression(theta)
        result[(theta)] = evaluateModel(ln, True)

    pp.pprint(result)
    resultAnalysis(result)

def hiddenOptimizeNeural():
    # Get best number of hidden nodes for one-layer NN
    result = {}
    for i in range(4, 12):
        nn = neuralNetwork(i)
        for eta in range(500, 2501, 100):
            batch = 150
            res = evaluateModel(nn, True, eta, batch)
            result[(i, eta, batch)] = res
            print "=============================="
            print i, eta, batch, res
            print "=============================="
    
    pp.pprint(result)
    resultAnalysis(result)

# hiddenOptimizeNeural()
# lr = linearRegression(0.01)
nn = neuralNetwork(15)
# for i in range(5):
# res_lr = evaluateModel(lr, False, 600, 150)
res_nn = evaluateModel(nn, False, 800, 150)
# print "LR: ", res_lr
print "NN: ", res_nn
