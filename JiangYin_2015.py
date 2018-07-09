import sys
import custom_model as cm
import numpy as np
import random
from sklearn.metrics.classification import accuracy_score, recall_score, f1_score
import scipy.stats as st

from keras.layers import Input, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, Activation, AveragePooling2D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from keras.models import Model
from keras import backend as K
K.set_image_data_format('channels_first')

def custom_model1(inp, n_classes):
    #Original architecture
    H = Conv2D(filters=5, kernel_size=(5, 5))(inp)
    H = Activation('relu')(H)
    H = AveragePooling2D(pool_size=(4, 4))(H)

    H = Conv2D(filters=10, kernel_size=(5, 5))(H)
    H = Activation('relu')(H)
    H = AveragePooling2D(pool_size=(2, 2))(H)

    H = Flatten()(H)
    H = Dense(120)(H)
    H = Activation('relu')(H)

    H = Dense(n_classes)(H)
    H = Activation('softmax')(H)

    model = Model([inp], H)

    return model

def custom_model2(inp, n_classes):
    # Adapted architecture
    H = Conv2D(filters=5, kernel_size=(4, 4), padding='same')(inp)
    H = Activation('relu')(H)
    H = AveragePooling2D(pool_size=(2, 2))(H)

    H = Conv2D(filters=10, kernel_size=(4, 4), padding='same')(H)
    H = Activation('relu')(H)
    H = AveragePooling2D(pool_size=(2, 2))(H)

    H = Flatten()(H)
    H = Dense(120)(H)
    H = Activation('relu')(H)

    H = Dense(n_classes)(H)
    H = Activation('softmax')(H)

    model = Model([inp], H)

    return model

def check_neighboring(sis, elem1, elem2):
    if(len(sis)==1):
        return False
    for i in range(0, len(sis)-1):
        if sis[i] == elem1 and sis[i+1] == elem2:
            return True
        if sis[i] == elem2 and sis[i + 1] == elem1:
            return True
    return False

def activity_image(raw_signals):
    seq =  np.arange(0, raw_signals.shape[3], 1)
    sis = []
    n_sis = 1
    i = 0
    j = i+1
    sis.append(i)
    while i!=j:
        if j==len(seq):
            j=0
        elif check_neighboring(sis, i, j) == False:
            sis.append(j)
            i = j
            j = j+1
        else:
            j = j + 1

    output = []
    for sample in raw_signals:
        signal_image = sample[0]
        signal_image = signal_image[:, sis]
        signal_image = np.transpose(signal_image)

        fshift = np.fft.fftshift(signal_image)
        fshift = np.transpose(fshift)
        # import cv2
        # cv2.imshow('tete', magnitude_spectrum)
        # cv2.waitKey(1000)

        output.append([fshift])

    output = np.array(output)
    return output

if __name__ == '__main__':
    #Paper: Human Activity Recognition using Wearable Sensors by Deep Convolutional Neural Networks
    np.random.seed(12227)

    if (len(sys.argv) > 1):
        data_input_file = sys.argv[1]
    else:
        data_input_file = 'data/LOSO/MHEALTH.npz'

    tmp = np.load(data_input_file)
    X = tmp['X']
    y = tmp['y']
    folds = tmp['folds']

    n_class = y.shape[1]

    X = activity_image(X)
    _, _, img_rows, img_cols = X.shape
    avg_acc = []
    avg_recall = []
    avg_f1 = []

    print('Jiang and Yin 2015 {}'.format(data_input_file))

    for i in range(0, len(folds)):
        train_idx = folds[i][0]
        test_idx = folds[i][1]

        X_train = X[train_idx]
        X_test = X[test_idx]

        inp = Input((1, img_rows, img_cols))
        #model = custom_model1(inp, n_classes=n_class)
        model = custom_model2(inp, n_classes=n_class)

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

    ic_acc = st.t.interval(0.9, len(avg_acc) - 1, loc=np.mean(avg_acc), scale=st.sem(avg_acc))
    ic_recall = st.t.interval(0.9, len(avg_recall) - 1, loc=np.mean(avg_recall), scale=st.sem(avg_recall))
    ic_f1 = st.t.interval(0.9, len(avg_f1) - 1, loc=np.mean(avg_f1), scale=st.sem(avg_f1))
    print('Mean Accuracy[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_acc), ic_acc[0], ic_acc[1]))
    print('Mean Recall[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_recall), ic_recall[0], ic_recall[1]))
    print('Mean F1[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_f1), ic_f1[0], ic_f1[1]))
