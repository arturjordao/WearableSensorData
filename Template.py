import numpy as np
from sklearn.metrics.classification import accuracy_score, recall_score, f1_score
import scipy.stats as st
import sys

if __name__ == '__main__':
    np.random.seed(12227)

    if (len(sys.argv) > 1):
        data_input_file = sys.argv[1]
    else:
        data_input_file = '<>.npz'

    tmp = np.load(data_input_file)
    X = tmp['X']
    #For sklearn methods X = X[:, 0, :, :]
    y = tmp['y']
    folds = tmp['folds']

    n_class = y.shape[1]

    avg_acc = []
    avg_recall = []
    avg_f1 = []
    y = np.argmax(y, axis=1)

    print('Template 2017 {}'.format(data_input_file))

    for i in range(0, len(folds)):
        train_idx = folds[i][0]
        test_idx = folds[i][1]

        X_train = X[train_idx]
        X_test = X[test_idx]

        X_train = feature_extraction(X_train)
        X_test = feature_extraction(X_test)

        #Your train goes here. For instance:
        method.fit(X_train, y[train_idx])

        #Your testing goes here. For instance:
        y_pred = method.predict(X_test)

        acc_fold = accuracy_score(y[test_idx], y_pred)
        avg_acc.append(acc_fold)

        recall_fold = recall_score(y[test_idx], tmp, average='macro')
        avg_recall.append(recall_fold)

        f1_fold  = f1_score(y[test_idx], tmp, average='macro')
        avg_f1.append(f1_fold)

        print('Accuracy[{:.4f}] Recall[{:.4f}] F1[{:.4f}] at fold[{}]'.format(acc_fold, recall_fold, f1_fold ,i))
        print('______________________________________________________')

    ic_acc = st.t.interval(0.9, len(avg_acc) - 1, loc=np.mean(avg_acc), scale=st.sem(avg_acc))
    ic_recall = st.t.interval(0.9, len(avg_recall) - 1, loc=np.mean(avg_recall), scale=st.sem(avg_recall))
    ic_f1 = st.t.interval(0.9, len(avg_f1) - 1, loc=np.mean(avg_f1), scale=st.sem(avg_f1))
    print('Mean Accuracy[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_acc), ic_acc[0], ic_acc[1]))
    print('Mean Recall[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_recall), ic_recall[0], ic_recall[1]))
    print('Mean F1[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_f1), ic_f1[0], ic_f1[1]))