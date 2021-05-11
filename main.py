import pandas as pd
import pickle
import os

from baseline import Baseline
from preprocessor import Preprocessor
from constant import Constant

def save(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    train = pd.read_csv('data/raw/train.csv')[:Constant.MAX_DATA]
    model_name = 'baseline.pkl'

    preprocessor = Preprocessor()
    baseline = Baseline()

    print('[LOG] Preprocessing Data')
    preprocessor.preprocess()

    print('[LOG] Training')
    baseline.train(train)

    print('[LOG] Saving Model')
    save(baseline, os.path.join(Constant.MODEL_PATH, model_name))