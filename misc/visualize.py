import util
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pprint
pp = pprint.PrettyPrinter()

# titles_train, X_train, Y_train, titles_test, X_test, Y_test = util.getCachedDataset("data/nn_features")

def visualizeScores(scores):
    # Provides histogram of scores
    hist, bins, patches = plt.hist(scores, bins=50, edgecolor='black', linewidth=2)
    for p in patches:
        p.set_color('#F06666')
    # plt.savefig(name)
    plt.show()

# Y_total = Y_train + Y_test
# Y_perc_total = [stats.percentileofscore(Y_total, a, 'rank') / 100.0 for a in Y_total]
# Y_perc_train = [stats.percentileofscore(Y_total, a, 'rank') / 100.0 for a in Y_train]
# Y_perc_test = [stats.percentileofscore(Y_total, a, 'rank') / 100.0 for a in Y_test]

# visualizeScores("images/Y_total.jpg", Y_total)
# visualizeScores("images/Y_perc.jpg", Y_perc_total)

def scatterPlot(feat, x_scat_train, y_scat_train, x_scat_test, y_scat_test):
    fig, ax = plt.subplots()
    colors = [(0.608, 0.608, 0.608, 0.3)] * len(x_scat_train) + [(0.89, 0.247, 0.392, 0.8)] * len(x_scat_test)
    ax.scatter(x_scat_train + x_scat_test, y_scat_train + y_scat_test, 
            c=colors, lw=0)
    plt.xlabel(feat)
    plt.ylabel('Normalized score')
    plt.savefig("images/" + feat + ".jpg")

def allFeatures():
    features = []
    for d in X_train + X_test:
        for feat in d:
            features.append(feat)
    return features

def analyzeFeatures():
    for feat in allFeatures():
        if feat in ["DET", "anger", "richness"]:
            x_scat_train = []
            y_scat_train = []
            for i, d in enumerate(X_train):
                if feat in d:
                    x_scat_train.append(d[feat])
                    y_scat_train.append(Y_perc_train[i])

            x_scat_test = []
            y_scat_test = []
            for i, d in enumerate(X_test):
                if feat in d:
                    x_scat_test.append(d[feat])
                    y_scat_test.append(Y_perc_test[i])

            scatterPlot(feat, x_scat_train, y_scat_train, x_scat_test, y_scat_test)

# analyzeFeatures()
