from sklearn.metrics.classification import accuracy_score, recall_score, f1_score
import scipy.stats as st
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
import random
import copy

from keras.layers import Input, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, Activation
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback, LearningRateScheduler
from keras.models import Model

import sys
import custom_model as cm

class LatentHyperNet(BaseEstimator, ClassifierMixin):
    __name__ = 'Latent Hyper Net'

    def __init__(self, n_iter=1500, eps=1e-6, n_comp=2, mode='regression', dm_method=None, model=None, layers=None):
        self.n_iter = n_iter
        self.eps = eps
        self.n_comp = n_comp
        self.mode = mode
        self.dm_layer = []
        self.dm_method = dm_method
        self.model = self.custom_model(model=model, layers=layers)
        self.layers = layers

    def custom_model(self, model, layers):
        input_shape = model.input_shape
        input_shape = (input_shape[1], input_shape[2], input_shape[3])
        inp = Input(input_shape)
        feature_maps = [Model(model.input, model.get_layer(index=i).output)(inp) for i in layers]
        model = Model(inp, feature_maps)
        return model

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError()

        #self.classes_, target = np.unique(y, return_inverse=True)
        target = y
        target[target == 0] = -1
        if self.dm_method == 'lda':
            target = np.argmax(target, axis=1)

        X = self.extract_features(X)

        if self.dm_method == 'pls':
            dm = PLSRegression(n_components=self.n_comp, scale=True, max_iter=self.n_iter, tol=self.eps)
        elif self.dm_method == 'pca':
            dm = PCA(self.n_comp)
        elif self.dm_method == 'lda':
            dm = LinearDiscriminantAnalysis()

        for layer_idx in range(0, len(self.layers)):
            dm_ = copy.copy(dm)
            dm_.fit(X[layer_idx], target)
            self.dm_layer.append(dm_)
            del dm_

        return self

    def transform(self, x):
        import numpy as np
        proj_x = None

        x = self.extract_features(x)

        for layer_idx in range(0, len(self.layers)):
            if proj_x is None:
                proj_x = self.dm_layer[layer_idx].transform(x[layer_idx])
            else:
                proj_tmp = self.dm_layer[layer_idx].transform(x[layer_idx])
                proj_x = np.column_stack((proj_x, proj_tmp))

        return proj_x

    def extract_features(self, X, verbose=False):
        import time
        feat_layers = [[] for x in range(0, len(self.layers))]

        idx_sample = 0
        for sample in X:
            start = time.time()
            sample = np.expand_dims(sample, axis=0)
            feat = self.model.predict(sample)
            for layer in range(0, len(self.layers)):
                feat_layers[layer].append(np.reshape(feat[layer], -1))

            if verbose == True:
                print('Extracting features {}/{} Time[{}]'.format(idx_sample, len(X), time.time() - start))
            idx_sample = idx_sample + 1

        return feat_layers

    def get_features(self, X):
        features = self.extract_features(X)
        X = None
        for layer_idx in range(0, len(self.layers)):
            if X is None:
                X = features[layer_idx]
            else:
                X_tmp = features[layer_idx]
                X = np.column_stack((X, X_tmp))
        X = np.array(X)
        return X

def custom_model(inp, n_classes, dataset_name):
    activation = 'relu'
    if dataset_name == 'UTD-MHAD1_1s' or dataset_name =='UTD-MHAD2_1s':
        H = Conv2D(filters=24, kernel_size=(12, 2))(inp)
        H = Activation(activation)(H)
        H = MaxPooling2D(pool_size=(2, 1))(H)

        H = Conv2D(filters=36, kernel_size=(12, 2))(H)
        H = Activation(activation)(H)
        H = MaxPooling2D(pool_size=(2, 1))(H)

        H = Flatten()(H)
        H = Dense(n_classes)(H)
        H = Activation('softmax')(H)

        model = Model([inp], H)
    else:
        H = Conv2D(filters=24, kernel_size=(12, 1))(inp)
        H = Activation(activation)(H)
        H = MaxPooling2D(pool_size=(2, 1))(H)

        H = Conv2D(filters=32, kernel_size=(12, 1))(H)
        H = Activation(activation)(H)
        H = MaxPooling2D(pool_size=(2, 1))(H)

        H = Conv2D(filters=40, kernel_size=(6, 1))(H)
        H = Activation(activation)(H)
        H = MaxPooling2D(pool_size=(2, 1))(H)

        H = Conv2D(filters=48, kernel_size=(2, 1))(H)
        H = Activation(activation)(H)
        H = MaxPooling2D(pool_size=(2, 1))(H)

        H = Flatten()(H)
        H = Dense(n_classes)(H)
        H = Activation('softmax')(H)

        model = Model([inp], H)

    return model

if __name__ == '__main__':
    # Paper: Latent HyperNet: Exploring the Layers of Convolutional Neural Networks
    np.random.seed(12227)

    if (len(sys.argv) > 1):
        data_input_file = sys.argv[1]
    else:
        data_input_file = '//storage.vpr.dcc.ufmg.br/home/projects/sensor2.0/SavedFeatures/LOSO/UTD-MHAD2_1s.npz'

    dataset_name = data_input_file.split('/')
    dataset_name = dataset_name[-1].replace('.npz', '')

    if dataset_name == 'UTD-MHAD1_1s' or dataset_name == 'UTD-MHAD2_1s':
        layers = [3, 6]
    else:
        layers = [3, 6, 9]

    tmp = np.load(data_input_file)
    X = tmp['X']
    y = tmp['y']
    folds = tmp['folds']

    n_class = y.shape[1]
    _, _, img_rows, img_cols = X.shape
    avg_acc = []
    avg_recall = []
    avg_f1 = []

    print('Jordao et al. 2018 {}'.format(data_input_file))
    for i in range(0, len(folds)):
        train_idx = folds[i][0]
        test_idx = folds[i][1]

        X_train = X[train_idx]
        X_test = X[test_idx]

        inp = Input((1, img_rows, img_cols))
        model = custom_model(inp, n_classes=n_class, dataset_name=dataset_name)

        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='Adadelta')
        model.fit(X_train, y[train_idx], batch_size=cm.bs, epochs=cm.n_ep,
                  verbose=0, callbacks=[cm.custom_stopping(value=cm.loss, verbose=2)],
                  validation_data=(X_train, y[train_idx]))


        hyper_net = LatentHyperNet(n_comp=19, model=model, layers=layers, dm_method='pls')
        hyper_net.fit(X_train, y[train_idx])
        X_train = hyper_net.transform(X_train)
        X_test = hyper_net.transform(X_test)

        inp = Input((X_train.shape[1],))
        fc = Dense(n_class)(inp)
        model = Activation('softmax')(fc)
        model = Model(inp, model)

        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='Adadelta')
        callbacks = [cm.custom_stopping(value=cm.loss, verbose=2)]

        model.fit(X_train, y[train_idx], batch_size=len(X_train),
                  epochs=4*cm.n_ep,#The drawback of the method, it require more iterations to converge (to the loss achieve cm.loss)
                   verbose=0, callbacks=callbacks, validation_data=(X_train, y[train_idx]))

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