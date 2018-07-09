import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics.classification import accuracy_score, recall_score, f1_score
import scipy.stats as st
import sys

import copy

def COR(sample):
    feat = []
    for axis_i in range(0, sample.shape[1]):
        for axis_j in range(axis_i+1, sample.shape[1]):
            cor = np.corrcoef(sample[:,axis_i], sample[:,axis_j])
            feat.append(cor[0][1])

    return feat

def MEAN(sample):
    feat = []
    for col in range(0, sample.shape[1]):
        std = np.mean(sample[:, col])
        feat.append(std)

    return feat

def feature_extraction(X):
    #Extracts the features, as mentioned by Catal et al. 2015
    # Mean - MEAN,
    # Correlation - COR
    X_tmp = []
    for sample in X:
        features = COR(copy.copy(sample))
        features = np.hstack((features, MEAN(copy.copy(sample))))
        X_tmp.append(features)

    X = np.array(X_tmp)
    return X

def train_boosting(X, y):
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(n_estimators=100)#Default param of weka
    clf.fit(X, y)
    return clf

if __name__ == '__main__':
    #Paper: Eating Activity Recognition for Health and Wellness: A Case Study on Asian Eating Style
    np.random.seed(12227)

    if (len(sys.argv) > 1):
        data_input_file = sys.argv[1]
    else:
        data_input_file = 'data/LOSO/MHEALTH.npz'

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
    print('Kim and Choi 2013 {}'.format(data_input_file))

    for i in range(0, len(folds)):
        train_idx = folds[i][0]
        test_idx = folds[i][1]

        X_train = X[train_idx]
        X_test = X[test_idx]

        X_train = feature_extraction(X_train)
        X_test = feature_extraction(X_test)

        clf = train_boosting(X_train, y[train_idx])

        tmp = clf.predict(X_test)

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
