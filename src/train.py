import os
import pandas as pd
import pickle

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import tensorflow_addons as tfa

from src.process.preprocessor import Preprocessor
from src.process.cleaner import Cleaner

from src.util.constant import Constant

from src.model.baseline import Baseline
from src.model.callbacks import Checkpoint
from src.model.lstm import LSTMModel
from src.model.convolution import ConvModel

def save(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def train_model(architecture,
                folder,
                filename,
                x_train, 
                y_train, 
                x_test, 
                y_test, 
                total_words,
                input_length,
                n_class,
                epochs=10,
                batch_size=256):

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

    f1 = tfa.metrics.F1Score(n_class, 'macro')

    model = architecture(total_words, input_length, n_class)
    model.build((None, x_train.shape[1]))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy', f1])

    print(model.summary())

    filepath = os.path.join(Constant.MODEL_PATH, folder, filename)
    callbacks = [Checkpoint(filepath)]

    history = model.fit(x_train, 
                        y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        callbacks=callbacks,
                        verbose=1)
    return model


def train(baseline=False,
          lstm=False,
          convolution=False):
    train = pd.read_csv('data/raw/train.csv')
    val = pd.read_csv('data/raw/val.csv')

    if baseline:
        train = train[:Constant.MAX_DATA]
        model_name = 'baseline.pkl'
        baseline = Baseline()
        baseline.train(train)
        print('[LOG] Saving Model')
        save(baseline, os.path.join(Constant.MODEL_PATH, model_name))
        return

    preprocessor_name = 'preprocessor.pkl'

    preprocessor = Preprocessor()
    cleaner = Cleaner()
    
    print('[LOG] Cleaning Data')
    train = cleaner.clean(train)
    val = cleaner.clean(val)

    train.to_csv(os.path.join(Constant.DATA_PATH, 'processed', 'train.csv'), index=False)
    train.to_csv(os.path.join(Constant.DATA_PATH, 'processed', 'val.csv'), index=False)

    print('[LOG] Preprocessing Data')
    x_train, y_train = preprocessor.fit_transform(train)
    x_test, y_test = preprocessor.transform(val, meta=True)
    save(preprocessor, os.path.join(Constant.PREPROCESSOR_PATH, preprocessor_name))

    print(f'Training {x_train.shape}')
    print(f'Validating {x_test.shape}')
    print(f'Train Label {y_train.shape}')
    print(f'Validation label {y_test.shape}')

    print('[LOG] Training')
    total_words = len(preprocessor.tokenizer.word_index) + 1
    input_length = preprocessor.maxlen
    n_class = preprocessor.n_class

    if convolution:
        model = ConvModel
        folder = 'convolution'
        filename = 'Conv1D'

    elif lstm:
        model = LSTMModel
        folder = 'lstm'
        filename = 'LSTM'

    train_model(model, 
                folder, 
                filename, 
                x_train, 
                y_train, 
                x_test, 
                y_test, 
                total_words, 
                input_length, 
                n_class)