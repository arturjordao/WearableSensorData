import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics.classification import accuracy_score, recall_score, f1_score
import scipy.stats as st
import sys

import copy

def A(sample):
    feat = []
    for col in range(0,sample.shape[1]):
        average = np.average(sample[:,col])
        feat.append(average)

    return feat

def SD(sample):
    feat = []
    for col in range(0, sample.shape[1]):
        std = np.std(sample[:, col])
        feat.append(std)

    return feat

def AAD(sample):
    feat = []
    for col in range(0, sample.shape[1]):
        data = sample[:, col]
        add = np.mean(np.absolute(data - np.mean(data)))
        feat.append(add)

    return feat

def ARA(sample):
    #Average Resultant Acceleration[1]:
    # Average of the square roots of the sum of the values of each axis squared √(xi^2 + yi^2+ zi^2) over the ED
    feat = []
    sum_square = 0
    sample = np.power(sample, 2)
    for col in range(0, sample.shape[1]):
        sum_square = sum_square + sample[:, col]

    sample = np.sqrt(sum_square)
    average = np.average(sample)
    feat.append(average)
    return feat

def TBP(sample):
    from scipy import signal
    feat = []
    sum_of_time = 0
    for col in range(0, sample.shape[1]):
        data = sample[:, col]
        peaks = signal.find_peaks_cwt(data, np.arange(1,4))

        feat.append(peaks)

    return feat

def feature_extraction(X):
    #Extracts the features, as mentioned by Catal et al. 2015
    # Average - A,
    # Standard Deviation - SD,
    # Average Absolute Difference - AAD,
    # Average Resultant Acceleration - ARA(1),
    # Time Between Peaks - TBP
    X_tmp = []
    for sample in X:
        features = A(copy.copy(sample))
        features = np.hstack((features, A(copy.copy(sample))))
        features = np.hstack((features, SD(copy.copy(sample))))
        features = np.hstack((features, AAD(copy.copy(sample))))
        features = np.hstack((features, ARA(copy.copy(sample))))
        #features = np.hstack((features, TBP(sample)))
        X_tmp.append(features)

    X = np.array(X_tmp)
    return X

def train_mlp(X, y):
    from sklearn.neural_network import MLPClassifier
    a = int((X.shape[1] + np.amax(y)) / 2 )#Default param of weka, amax(y) gets the number of classes
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (a,),
                        learning_rate_init=0.3, momentum=0.2, max_iter=500, #Default param of weka
                        )
    clf.fit(X, y)
    return clf

if __name__ == '__main__':
    #Paper: Activity recognition using cell phone accelerometers
    np.random.seed(12227)

    if (len(sys.argv) > 1):
        data_input_file = sys.argv[1]
    else:
        data_input_file = 'C:/Users/Artur Jordao/Desktop/UFMG/Projeto/sensors2.0/SavedFeatures/LOSO_10/UTD-MHAD2_1s.npz'

    tmp = np.load(data_input_file)
    X = tmp['X']
    X = X[:, 0, :, :]
    y = tmp['y']
    folds = tmp['folds']

    n_class = y.shape[1]

    avg_acc = []
    avg_recall = []
    avg_f1 = []
    y = np.argmax(y, axis=1)
    print('Kwapisz et al. 2010 {}'.format(data_input_file))

    for i in range(0, len(folds)):
        train_idx = folds[i][0]
        test_idx = folds[i][1]

        X_train = X[train_idx]
        X_test = X[test_idx]

        X_train = feature_extraction(X_train)
        X_test = feature_extraction(X_test)

        mlp = train_mlp(X_train, y[train_idx])

        tmp = mlp.predict(X_test)

        acc_fold = accuracy_score(y[test_idx], tmp)
        avg_acc.append(acc_fold)

        recall_fold = recall_score(y[test_idx], tmp, average='macro')
        avg_recall.append(recall_fold)

        f1_fold = f1_score(y[test_idx], tmp, average='macro')
        avg_f1.append(f1_fold)

        print('Accuracy[{:.4f}] Recall[{:.4f}] F1[{:.4f}] at fold[{}]'.format(acc_fold, recall_fold, f1_fold, i))
        print('______________________________________________________')

    ic_acc = st.t.interval(0.9, len(avg_acc) - 1, loc=np.mean(avg_acc), scale=st.sem(avg_acc))
    ic_recall = st.t.interval(0.9, len(avg_recall) - 1, loc=np.mean(avg_recall), scale=st.sem(avg_recall))
    ic_f1 = st.t.interval(0.9, len(avg_f1) - 1, loc=np.mean(avg_f1), scale=st.sem(avg_f1))
    print('Mean Accuracy[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_acc), ic_acc[0], ic_acc[1]))
    print('Mean Recall[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_recall), ic_recall[0], ic_recall[1]))
    print('Mean F1[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_f1), ic_f1[0], ic_f1[1]))