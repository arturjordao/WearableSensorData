import numpy as np
from keras.layers import Input, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, Activation
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from keras.models import Model
from keras import backend as K
K.set_image_data_format('channels_first')
n_ep = 200
loss = 0.2
bs = 1000

def custom_stopping(value=0.5, verbose=0):
    early = EarlyStoppingByLossVal(monitor='val_loss', value=value, verbose=verbose)
    return early

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_acc', value=0.95, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        # if current is None:
        # warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True