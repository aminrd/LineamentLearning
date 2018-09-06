
# Class : Model
import numpy as np
import random

# Importing Keras Libraries:
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


from globalVariables import *
from Utility import *


if DEBUG_MODE:
    print("### Importing MODEL Class ###")


def get_RotateNet(w):
    model = Sequential()
    model.add(Conv2D(8, 3, padding='valid', activation='relu', input_shape=(w, w, Layers)))
    model.add(Flatten())
    model.add(Dense(300, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model


class MODEL:
    """Class Model: define model here. Can train, predict and load previously trained parameters"""

    def __init__(self, w = WindowSize, param_dir = 0, checkpoint = CB + 'Default.hdf5'):
        self.model = get_RotateNet(w)

        if param_dir != 0:
            self.model.load_weights(param_dir)
        else:
            self.model_checkpoint = ModelCheckpoint(checkpoint, monitor='loss', verbose=DEBUG_MODE, save_best_only=False)

    def train(self, X, Y, epochs = 2):
        self.model.fit(X, Y, shuffle=True, batch_size=32, epochs=epochs, verbose=DEBUG_MODE, callbacks=[self.model_checkpoint])

    def predict(self, X):
        return self.model.predict(X, verbose = DEBUG_MODE)

