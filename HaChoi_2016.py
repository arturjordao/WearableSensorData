import sys
import custom_model as cm
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics.classification import accuracy_score, recall_score, f1_score
import scipy.stats as st

from keras.layers import Input, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, Activation, Concatenate, merge
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from keras.models import Model
from keras import backend as K
K.set_image_data_format('channels_first')

def custom_model(X_shape, idx_modalities, n_classes):
    # Architecture for PAMAP and MHEALTH datasets

    img_cols1 = idx_modalities[0]
    img_cols2 = idx_modalities[1] - idx_modalities[0]
    img_cols3 = idx_modalities[2] - idx_modalities[1]
    img_cols4 = X.shape[3] - idx_modalities[2]

    _, _, img_rows, img_cols = X.shape
    inp_modality1 = Input((1, img_rows, img_cols1))
    inp_modality2 = Input((1, img_rows, img_cols2))
    inp_modality3 = Input((1, img_rows, img_cols3))
    inp_modality4 = Input((1, img_rows, img_cols4))

    H1 = Conv2D(filters=5, kernel_size=(5, 5))(inp_modality1)
    H1 = Activation('relu')(H1)
    H1 = MaxPooling2D(pool_size=(4, 4))(H1)

    H2 = Conv2D(filters=5, kernel_size=(5, 5))(inp_modality2)
    H2 = Activation('relu')(H2)
    H2 = MaxPooling2D(pool_size=(4, 4))(H2)

    H3 = Conv2D(filters=5, kernel_size=(5, 5))(inp_modality3)
    H3 = Activation('relu')(H3)
    H3 = MaxPooling2D(pool_size=(4, 4))(H3)

    H4 = Conv2D(filters=5, kernel_size=(5, 3))(inp_modality4)#For PAMAP
    #H4 = Conv2D(filters=5, kernel_size=(5, 2))(inp_modality4)  # For MEHEALTH
    H4 = Activation('relu')(H4)
    H4 = MaxPooling2D(pool_size=(4, 1))(H4)

    shape_1 = int(H2.shape[1].value)
    shape_2 = int(H2.shape[2].value)
    shape_3 = int(H2.shape[3].value)
    inp_zeros = Input((shape_1, shape_2, shape_3))  # Here is the features map shape

    H = merge([H1, inp_zeros, H2, inp_zeros, H3, inp_zeros, H4], mode='concat', concat_axis=3)

    H = Conv2D(filters=10, kernel_size=(5, 5))(H)
    H = Activation('relu')(H)
    H = MaxPooling2D(pool_size=(2, 2))(H)

    H = Flatten()(H)
    H = Dense(120)(H)
    H = Activation('relu')(H)

    H = Dense(n_classes)(H)
    H = Activation('softmax')(H)

    model = Model([inp_modality1, inp_modality2, inp_modality3, inp_modality4, inp_zeros], H)

    return model, (shape_1, shape_2, shape_3)

def zero_padding_PAMAP(X):
    # Groups the heterogenous sensors for PAMAP
    idx_modalities = []
    idx_acc = [1, 2, 3, 4, 5, 6, 14, 15, 16, 17, 18, 19, 27, 28, 29, 30, 31, 32]
    idx_gyro = [7, 8, 9, 20, 21, 22, 33, 34, 35]
    idx_mag = [10, 11, 12, 23, 24, 25, 36, 37, 38]
    idx_temp = [0, 13, 26]
    X_acc = X[:, :, :, idx_acc]
    X_gyro = X[:, :, :, idx_gyro]
    X_mag = X[:, :, :, idx_mag]
    X_temp = X[:, :, :, idx_temp]
    X_zeros = np.zeros((X.shape[0], X.shape[1], X.shape[2], 2))  # Vertical Kernel-1 = 2

    X = X_acc
    X = np.concatenate((X, X_zeros), axis=3)
    idx_modalities.append(X.shape[3])

    X = np.concatenate((X, X_gyro), axis=3)
    X = np.concatenate((X, X_zeros),axis=3)
    idx_modalities.append(X.shape[3])

    X = np.concatenate((X, X_mag),axis=3)
    X = np.concatenate((X, X_zeros),axis=3)
    idx_modalities.append(X.shape[3])
    X = np.concatenate((X, X_temp),axis=3)

    return X, idx_modalities

def zero_padding_MHEALTH(X):
    # Groups the heterogenous sensors for MHEALTH
    idx_modalities = []
    idx_acc = [0, 1, 2, 5, 6, 7, 14, 15, 16]
    idx_gyro = [8, 9, 10, 17, 18, 19]
    idx_mag = [11, 12, 13, 20, 21, 22]
    idx_ele = [3, 4]
    X_acc = X[:, :, :, idx_acc]
    X_gyro = X[:, :, :, idx_gyro]
    X_mag = X[:, :, :, idx_mag]
    X_ele = X[:, :, :, idx_ele]
    X_zeros = np.zeros((X.shape[0], X.shape[1], X.shape[2], 2))  # Vertical Kernel-1 = 2

    X = X_acc
    X = np.concatenate((X, X_zeros), axis=3)
    idx_modalities.append(X.shape[3])

    X = np.concatenate((X, X_gyro), axis=3)
    X = np.concatenate((X, X_zeros),axis=3)
    idx_modalities.append(X.shape[3])

    X = np.concatenate((X, X_mag),axis=3)
    X = np.concatenate((X, X_zeros),axis=3)
    idx_modalities.append(X.shape[3])
    X = np.concatenate((X, X_ele),axis=3)

    return X, idx_modalities

def split_X(X, idx_modalities, zeros):
    X_tmp = []
    X_tmp.append(X[:, :, :, 0:idx_modalities[0]])
    X_tmp.append(X[:, :, :, idx_modalities[0]:idx_modalities[1]])
    X_tmp.append(X[:, :, :, idx_modalities[1]:idx_modalities[2]])
    X_tmp.append(X[:, :, :, idx_modalities[2]:X.shape[3]])
    X_tmp.append(zeros)
    return X_tmp

if __name__ == '__main__':
    #Paper: Multi-modal convolutional neural networks for activity recognition
    np.random.seed(12227)

    if (len(sys.argv) > 1):
        data_input_file = sys.argv[1]
    else:
        data_input_file = '//storage.vpr.dcc.ufmg.br/home/projects/sensor2.0/SavedFeatures/CV_0.5/PAMAP2P.npz'

    tmp = np.load(data_input_file)
    X = tmp['X']
    y = tmp['y']
    folds = tmp['folds']

    n_class = y.shape[1]

    #####Groups the heterogenous sensors for PAMAP2P##################
    #X, idx_modalities = zero_padding_PAMAP(X)
    X, idx_modalities = zero_padding_MHEALTH(X)

    _, _, img_rows, img_cols = X.shape
    avg_acc = []
    avg_recall = []
    avg_f1 = []

    print('Ha and Choi 2016 {}'.format(data_input_file))
    for i in range(0, len(folds)):
        train_idx = folds[i][0]
        test_idx = folds[i][1]

        X_train = X[train_idx]
        X_test = X[test_idx]

        model, inp_zeros = custom_model(X.shape, idx_modalities, n_classes=n_class)

        zeros_mat = np.zeros((X_train.shape[0], inp_zeros[0], inp_zeros[1], inp_zeros[2]))
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='Adadelta')

        X_train = split_X(X_train, idx_modalities, zeros_mat)

        model.fit(X_train, y[train_idx], batch_size=cm.bs, epochs=cm.n_ep,
                  verbose=0, callbacks=[cm.custom_stopping(value=cm.loss, verbose=1)], validation_data=(X_train, y[train_idx]))

        X_test = split_X(X_test, idx_modalities, zeros_mat)

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