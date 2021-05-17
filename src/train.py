import os
import pandas as pd
import pickle

from src.model.baseline import Baseline
from src.model.convolution import train_model
from src.process.preprocessor import Preprocessor
from src.process.cleaner import Cleaner

from src.util.constant import Constant


def save(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def train(baseline=False):
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

    train.to_csv(os.path.join(Constant.DATA_PATH,
                              'processed', 'train.csv'), index=False)
    train.to_csv(os.path.join(Constant.DATA_PATH,
                              'processed', 'val.csv'), index=False)

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

    # train_model(x_train, y_train, x_test, y_test,
    #             total_words, input_length, n_class)
