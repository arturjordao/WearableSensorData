import sys
import custom_model as cm
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics.classification import accuracy_score, recall_score, f1_score
import scipy.stats as st

from keras.layers import Input, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, Activation, AveragePooling2D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from keras.models import Model
from keras import backend as K
K.set_image_data_format('channels_first')

def custom_model(inp, n_classes):
    activation = 'relu'

    H = Conv2D(filters=18, kernel_size=(12, 2))(inp)
    H = Activation(activation)(H)
    H = MaxPooling2D(pool_size=(2, 1))(H)

    H = Conv2D(filters=36, kernel_size=(13, 1))(H)
    H = Activation(activation)(H)
    H = MaxPooling2D(pool_size=(2, 1))(H)

    H = Conv2D(filters=24, kernel_size=(12, 1))(H)
    H = Activation(activation)(H)
    H = MaxPooling2D(pool_size=(2, 1))(H)

    H = Flatten()(H)
    H = Dense(n_classes)(H)
    H = Activation('softmax')(H)

    model = Model([inp], H)

    return model

if __name__ == '__main__':
    #Paper: A Deep Learning Approach to Human Activity Recognition Based on Single Accelerometer
    np.random.seed(12227)

    if (len(sys.argv) > 1):
        data_input_file = sys.argv[1]
    else:
        data_input_file = '//storage.vpr.dcc.ufmg.br/home/projects/sensor2.0/SavedFeatures/CV_0.5/MHEALTH_5s.npz'

    tmp = np.load(data_input_file)
    X = tmp['X']
    y = tmp['y']
    folds = tmp['folds']

    n_class = y.shape[1]
    _, _, img_rows, img_cols = X.shape
    avg_acc = []
    avg_recall = []
    avg_f1 = []

    print('Chen and Xue 2015 {}'.format(data_input_file))

    for i in range(0, len(folds)):
        train_idx = folds[i][0]
        test_idx = folds[i][1]

        X_train = X[train_idx]
        X_test = X[test_idx]

        inp = Input((1, img_rows, img_cols))
        model = custom_model(inp, n_classes=n_class)

        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='Adadelta')
        model.fit(X_train, y[train_idx], batch_size=cm.bs, epochs=cm.n_ep,
                  verbose=0, callbacks=[cm.custom_stopping(value=cm.loss, verbose=1)], validation_data=(X_train, y[train_idx]))

        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)

        y_true = np.argmax(y[test_idx], axis=1)

        acc_fold = accuracy_score(y_true, y_pred)
        avg_acc.append(acc_fold)

        recall_fold = recall_score(y_true, y_pred, average='macro')
        avg_recall.append(recall_fold)

        f1_fold = f1_score(y_true, y_pred, average='macro')
        avg_f1.append(f1_fold)

        print('Accuracy[{:.4f}] Recall[{:.4f}] F1[{:.4f}] at fold[{}]'.format(acc_fold, recall_fold, f1_fold, i))
        print('______________________________________________________')
        del model

    ic_acc = st.t.interval(0.9, len(avg_acc) - 1, loc=np.mean(avg_acc), scale=st.sem(avg_acc))
    ic_recall = st.t.interval(0.9, len(avg_recall) - 1, loc=np.mean(avg_recall), scale=st.sem(avg_recall))
    ic_f1 = st.t.interval(0.9, len(avg_f1) - 1, loc=np.mean(avg_f1), scale=st.sem(avg_f1))
    print('Mean Accuracy[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_acc), ic_acc[0], ic_acc[1]))
    print('Mean Recall[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_recall), ic_recall[0], ic_recall[1]))
    print('Mean F1[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_f1), ic_f1[0], ic_f1[1]))