import tensorflow as tf
import numpy as np
import math
# Consistent testing
np.random.seed(7)
tf.set_random_seed(7)
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense
from scipy.sparse import vstack
from sklearn.model_selection import train_test_split
import pprint
pp = pprint.PrettyPrinter()
import features_nn
import visualize

print "Extracting features..."
# Temporal features 
titles_train, X_train, Y_train, titles_test, X_test, Y_test, X_all, Y_all = features_nn.getFeatures(True, "data/nn2_features", "")
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.3, random_state=3)
print "Done."

# Visualize stuff
# visualize.visualizeScores(Y_train)
# print sum(Y_train)/len(Y_train)

dim = X_train.shape[1]

# Train and Test set are randomly distributed: 0.6, 0.24, 0.16
X_rand_train, X_rand_test, Y_rand_train, Y_rand_test = train_test_split(X_all, Y_all, test_size=0.4, random_state=3)
X_rand_val, X_rand_test, Y_rand_val, Y_rand_test = train_test_split(X_test, Y_test, test_size=0.4, random_state=3)

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

def evaluateModel(m, val=False, eta=1000, batch=200):
    # Runs model given parameters
    print "Test = random"
    m.fit(X_rand_train.todense(), Y_rand_train, epochs=eta, batch_size=batch, 
            verbose=1, shuffle=False)
    if val:
        results_rand =  m.evaluate(X_rand_val.todense(), Y_rand_val, verbose=0)
    else:
        results_rand =  m.evaluate(X_rand_test.todense(), Y_rand_test, verbose=0)

    print "Test = 2015+"
    m.fit(X_train.todense(), Y_train, epochs=eta, batch_size=batch, 
            verbose=1, shuffle=False)
    if val:
        results =  m.evaluate(X_val.todense(), Y_val, verbose=0)
    else:
        results =  m.evaluate(X_test.todense(), Y_test, verbose=0)


    # prediction = m.predict(X_test.todense())
    # for i, predict in enumerate(prediction):
    #     if abs(predict - Y_test[i]) < 0.1:
    #         print "--------------------------------"
    #         print "Title:      ", titles_test[i]
    #         print "Prediction: ", predict
    #         print "Actual:     ", Y_test[i]
    #         print "--------------------------------"

    return results, results_rand

def resultAnalysis(result):
    # Given a dictionary of (params) => TEMP: (MSE, MAE), RAND: (MSE, MAE), get best of each
    best_mse_temp = sorted(result.iteritems(), key=lambda (k, v): v[0][0])[0]
    print "Best MSE Temp: ", best_mse_temp # (500, 160)
    best_mae_temp = sorted(result.iteritems(), key=lambda (k, v): v[0][1])[0]
    print "Best MAE Temp: ", best_mae_temp # (700, 110)
    best_temp = sorted(result.iteritems(), key=lambda (k, v): v[0][0] + v[0][1])[0]
    print "Best Overall Temp: ", best_temp

    best_mse_rand = sorted(result.iteritems(), key=lambda (k, v): v[1][0])[0]
    print "Best MSE Rand: ", best_mse_rand # (500, 160)
    best_mae_rand = sorted(result.iteritems(), key=lambda (k, v): v[1][1])[0]
    print "Best MAE Rand: ", best_mae_rand # (700, 110)
    best_rand = sorted(result.iteritems(), key=lambda (k, v): v[1][0] + v[1][1])[0]
    print "Best Overall Rand: ", best_rand

def hyperOptimize():
    # Get best eta and batch_size for linear regression using validation set
    ln = linearRegression(0)
    result = {}
    for eta in range(600, 1001, 100):
        for batch in range(200, 1001, 200):
            res, res_rand = evaluateModel(ln, True, eta, batch)
            result[(eta, batch)] = (res, res_rand)
            print eta, batch, res, res_rand
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
        res = evaluateModel(nn, True, 2000, 150)
        result[(i)] = res
        print "=============================="
        print i, res
        print "=============================="
    
    pp.pprint(result)
    resultAnalysis(result)

lr = linearRegression(0)
# nn = neuralNetwork()
res_lr, res_lr_rand = evaluateModel(lr, False, 800, 200)
# res_nn, res_nn_rand = evaluateModel(nn, False, 1000, 150)
print "LR Temporal Split: ", res_lr
print "LR Random Distrib: ", res_lr_rand
# print "NN Temporal Split: ", res_nn
# print "NN Random Distrib: ", res_nn_rand
