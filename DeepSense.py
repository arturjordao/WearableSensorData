import numpy as np
import keras
from sklearn.metrics.classification import accuracy_score, recall_score, f1_score
import scipy.stats as st
from scipy.fftpack import fft, ifft
import sys
import custom_model as cm
import tensorflow as tf
from keras import backend as K
from keras.layers.wrappers import TimeDistributed

K.set_image_data_format('channels_last')

SEPCTURAL_SAMPLES = 10
FEATURE_DIM = SEPCTURAL_SAMPLES * 6 * 2
CONV_LEN = 3
CONV_LEN_INTE = 3  # 4
CONV_LEN_LAST = 3  # 5
CONV_NUM = 64
CONV_MERGE_LEN = 8
CONV_MERGE_LEN2 = 6
CONV_MERGE_LEN3 = 4
CONV_NUM2 = 64
INTER_DIM = 120
OUT_DIM = 6  # len(idDict)
WIDE = 20
CONV_KEEP_PROB = 0.8

BATCH_SIZE = 64
TOTAL_ITER_NUM = 1000000000

# select = 'a'

# metaDict = {'a': [119080, 1193], 'b': [116870, 1413], 'c': [116020, 1477]}
# TRAIN_SIZE = metaDict[select][0]
# EVAL_DATA_SIZE = metaDict[select][1]
# EVAL_ITER_NUM = int(math.ceil(EVAL_DATA_SIZE / BATCH_SIZE))

conv_filters = 64

def toNHWCFormat(X_modas):
    aux_modas = []
    for i in range(len(X_modas)):
        aux = np.squeeze(X_modas[i])
        aux_modas.append(np.expand_dims(aux, axis=-1))

    return aux_modas
def reshapeArray(array, size_div):
    sensors_array = np.asarray(array)
    size_sensors = int(sensors_array.shape[0])
    list_reshaped = []
    for i in range(size_sensors):
        shape_array = sensors_array[i].shape
        sensor_array = sensors_array[i]
        print(shape_array)
        if int(shape_array[2] / size_div * size_div) != int(shape_array[2]):
            print('Erro: Size of the division invalid')
            return None
        else:
            array_reshaped = sensor_array.reshape(int(shape_array[0]), size_div, int(shape_array[1]), int(shape_array[2]/size_div), int(shape_array[3]))
            list_reshaped.append(array_reshaped)

    return list_reshaped


def fourier_transform_n_dim(array):
    sensors_array = np.asarray(array)
    size_sensors = int(sensors_array.shape[0])
    size_samples = int(sensors_array.shape[1])
    size_inter = int(sensors_array.shape[2])
    list_fourier = []
    for i in range(size_sensors):
        list_fourier_p3 = []
        for j in range(size_samples):
            list_fourier_p2 = []
            for b in range(size_inter):
                list_fourier_p = []
                sensor_data = sensors_array[i][j][b][0][:]
                list_fourier_p = np.fft.fftn(sensor_data)
                list_fourier_p2.append(list_fourier_p)
            list_fourier_p3.append(list_fourier_p2)
        list_fourier.append(list_fourier_p3)
    list_fourier = np.asarray(list_fourier)
    list_fourier = list_fourier.reshape(size_sensors, size_samples, size_inter, 1, sensors_array.shape[4], 3)
    return list_fourier

def _stream(inp, n_samples):
    # data_format='NHWC' means [batch, in_depth, in_height, in_width, in_channels]
    #conv1 = TimeDistributed(keras.layers.Conv2D(
        # filters=CONV_NUM, kernel_size=(1, 2 * 3 * CONV_LEN), activation=None, strides=[1, 2 * 3],
        # padding='VALID', data_format='channels_last'))(inp)
    conv1 = TimeDistributed(keras.layers.Conv2D(
        filters=CONV_NUM, kernel_size=(2 * 3 * CONV_LEN, 1), activation=None, strides=[2 * 3, 1],
        padding='VALID', data_format='channels_last'))(inp)
    conv1 = TimeDistributed(keras.layers.normalization.BatchNormalization())(conv1)
    conv1 = TimeDistributed(keras.layers.core.Activation('relu'))(conv1)

    conv1 = TimeDistributed(keras.layers.core.Dropout(CONV_KEEP_PROB))(conv1)
    #conv1 = TimeDistributed(keras.layers.core.Dropout(0.8))(conv1)


    # conv2 = TimeDistributed(keras.layers.Conv2D(
    #     filters=CONV_NUM, kernel_size=(1, CONV_LEN_INTE), activation=None, strides=[1, 1],
    #     padding='VALID', data_format='channels_last'))(conv1)
    conv2 = TimeDistributed(keras.layers.Conv2D(
        filters=CONV_NUM, kernel_size=(CONV_LEN_INTE, 1), activation=None, strides=[1, 1],
        padding='VALID', data_format='channels_last'))(conv1)
    conv2 = TimeDistributed(keras.layers.normalization.BatchNormalization())(conv2)
    conv2 = TimeDistributed(keras.layers.core.Activation('relu'))(conv2)
    #conv2 = TimeDistributed(keras.layers.core.Dropout(CONV_KEEP_PROB, noise_shape=[n_samples, 1, int(conv1.shape[3])]))(conv2)
    conv2 = TimeDistributed(keras.layers.core.Dropout(CONV_KEEP_PROB))(conv2)
    #conv2 = TimeDistributed(keras.layers.core.Dropout(0.8))(conv2)

    # conv3 = TimeDistributed(keras.layers.Conv2D(
    #     filters=CONV_NUM, kernel_size=(1, CONV_LEN_LAST), activation=None, strides=[1, 1],
    #     padding='VALID', data_format='channels_last'))(conv2)
    conv3 = TimeDistributed(keras.layers.Conv2D(
        filters=CONV_NUM, kernel_size=(CONV_LEN_LAST, 1), activation=None, strides=[1, 1],
        padding='VALID', data_format='channels_last'))(conv2)
    conv3 = TimeDistributed(keras.layers.normalization.BatchNormalization())(conv3)
    conv3 = TimeDistributed(keras.layers.core.Activation('relu'))(conv3)
    #conv3 = TimeDistributed(keras.layers.Reshape((1, int(conv3.shape[2]), int(conv3.shape[3]))))(conv3)

    return conv3

def _fusion(sensors_streams, n_samples):


    # axis=2 means concat on channels - [samples, timestep, channels, row, col]
    concat_layer = keras.layers.concatenate(sensors_streams, axis=2)
    #concat_shape = concat_layer.shape
    concat_layer = TimeDistributed(keras.layers.core.Dropout(CONV_KEEP_PROB))(concat_layer)
    #concat_layer = TimeDistributed(keras.layers.core.Dropout(0.8))(concat_layer)
    conv1 = TimeDistributed(keras.layers.Conv2D(
        filters=CONV_NUM2, kernel_size=(2, CONV_MERGE_LEN), activation=None, strides=[1, 1],
        padding='SAME', data_format='channels_last'))(concat_layer)
    conv1 = TimeDistributed(keras.layers.normalization.BatchNormalization())(conv1)
    conv1 = TimeDistributed(keras.layers.core.Activation('relu'))(conv1)
    #conv1_shape = conv1.shape
    #noise_shape=[n_samples, 1, 1, int(conv1_shape[3])]
    #conv1 = keras.layers.core.Dropout(0.8)(conv1)
    conv1 = TimeDistributed(keras.layers.core.Dropout(CONV_KEEP_PROB))(conv1)
    #conv1 = TimeDistributed(keras.layers.core.Dropout(0.8))(conv1)

    conv2 = TimeDistributed(keras.layers.Conv2D(
        filters=CONV_NUM2, kernel_size=(2, CONV_MERGE_LEN2), activation=None, strides=[1, 1],
        padding='SAME', data_format='channels_last'))(conv1)
    #data_format = 'channels_first'
    conv2 = TimeDistributed(keras.layers.normalization.BatchNormalization())(conv2)
    conv2 = TimeDistributed(keras.layers.core.Activation('relu'))(conv2)
   # conv2_shape = conv2.shape
    conv2 = TimeDistributed(keras.layers.core.Dropout(CONV_KEEP_PROB))(conv2)
    #conv2 = TimeDistributed(keras.layers.core.Dropout(0.8))(conv2)

    conv3 = TimeDistributed(keras.layers.Conv2D(
        filters=CONV_NUM2, kernel_size=(2, CONV_MERGE_LEN3), activation=None, strides=[1, 1],
        padding='SAME', data_format='channels_last'))(conv2)
    conv3 = TimeDistributed(keras.layers.normalization.BatchNormalization())(conv3)
    conv3 = TimeDistributed(keras.layers.core.Activation('relu'))(conv3)
    #output_shape = conv3.shape

    conv3 = TimeDistributed(keras.layers.Flatten())(conv3)
    #conv3 = TimeDistributed(
        # keras.layers.Reshape((n_samples, int(conv3.shape[2]) * int(conv3.shape[3]) * int(conv3.shape[4]))))(
        # conv3)

    return conv3

def _RNN(input):
    #esse dropout vai ser aplicado no input
    gru_cell1 = keras.layers.GRUCell(120, dropout=0.5)
    #Verificar dropout aqui
    gru_cell2 = keras.layers.GRUCell(120, dropout=0.5)
    #Verificar dropout aqui
    # 2688 eh o numero de dimensoes
   # input = keras.layers.Reshape((2688, 1))(input)
    rnn_layer = keras.layers.RNN([gru_cell1, gru_cell2], return_sequences=True)(input)
    dropout_output = TimeDistributed(keras.layers.core.Dropout(0.5))(rnn_layer)
    time_distributed_merge_layer = keras.layers.Lambda(function=lambda x: K.mean(x, axis=1),
                                                       output_shape=lambda shape: (shape[0],) + shape[2:])(dropout_output)

    #rnn_layer = keras.layers.Reshape((120, 1))(rnn_layer)
    #rnn_layer = keras.layers.Flatten()(rnn_layer)


    return time_distributed_merge_layer




def deepsense(input, n_class, n_samples):
    sensors_streams = []
    for inp in input:
        sensors_streams.append(_stream(inp, n_samples))
    merge = _fusion(sensors_streams, n_samples)
    out = _RNN(merge)

    out = keras.layers.Dense(n_class)(out)
    out = keras.layers.core.Activation('softmax')(out)

    # -------------- model buider  --------------------------------------------
    model = keras.models.Model(inputs=inputs, outputs=out)
    model.compile(loss='categorical_crossentropy',
                  optimizer='RMSProp',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    np.random.seed(12227)

    if len(sys.argv) > 1:
        data_input_file = sys.argv[1]
    else:
        data_input_file = 'Y:/Projects/fusion/sensor_fusion/SavedFeatures/LOSO/USCHAD.npz'

    tmp = np.load(data_input_file)
    X = tmp['X']
    # For sklearn methods X = X[:, 0, :, :]
    y = tmp['y']
    folds = tmp['folds']

    X_modas = []

    dataset_name = data_input_file.split('/')[-1]
    if dataset_name == 'UTD-MHAD2_1s.npz' or dataset_name == 'UTD-MHAD1_1s.npz' or dataset_name == 'USCHAD.npz':
        X_modas.append(X[:,:,:,0:3])
        X_modas.append(X[:, :, :, 3:6])
    elif dataset_name == 'WHARF.npz' or dataset_name == 'WISDM.npz':
        X_modas.append(X)
    elif dataset_name == 'MHEALTH.npz':
        X_modas.append(X[:, :, :, 0:3])
        X_modas.append(X[:, :, :, 3:6])
        X_modas.append(X[:, :, :, 6:9])
    elif dataset_name == 'PAMAP2P.npz':
        X_modas.append(X[:, :, :, 0:3])
        X_modas.append(X[:, :, :, 3:6])
        X_modas.append(X[:, :, :, 6:9])
        #X_modas.append(X[:, :, :, 9:10])


    n_class = y.shape[1]

    avg_acc = []
    avg_recall = []
    avg_f1 = []

    size_div = 10
    X_modas = reshapeArray(X_modas, size_div)
    X_modas = np.absolute(fourier_transform_n_dim(X_modas))
    X_modas = toNHWCFormat(X_modas)

    print('DeepSense Yao et al. 2017 {}'.format(data_input_file))


    for i in range(0, len(folds)):
        train_idx = folds[i][0]
        test_idx = folds[i][1]

        X_train = []
        X_test = []
        for x in X_modas:
            X_train.append(x[train_idx])
            X_test.append(x[test_idx])

        inputs = []

        for x in X_modas:
            inputs.append(keras.layers.Input((x.shape[1], x.shape[2], x.shape[3], x.shape[4])))
        _model = deepsense(inputs, n_class, X_modas[0].shape[0])
        print("Batch:{}, Num epochs: {}, Loss: {}").format(cm.bs, cm.n_ep, cm.loss)

        _model.fit(X_train, y[train_idx], cm.bs, cm.n_ep, verbose=0,
                   callbacks=[cm.custom_stopping(value=cm.loss, verbose=1)],
                   validation_data=(X_train, y[train_idx]))

        # Your testing goes here. For instance:
        y_pred = _model.predict(X_test)

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
    print('Mean Accuracy {:.4f}|[{:.4f}, {:.4f}]'.format(np.mean(avg_acc), ic_acc[0], ic_acc[1]))
    print('Mean Recall {:.4f}|[{:.4f}, {:.4f}]'.format(np.mean(avg_recall), ic_recall[0], ic_recall[1]))
    print('Mean F1 {:.4f}|[{:.4f}, {:.4f}]'.format(np.mean(avg_f1), ic_f1[0], ic_f1[1]))
