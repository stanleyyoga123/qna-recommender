import pandas as pd
import pickle
import os

from src.model.baseline import Baseline
from src.process.preprocessor import Preprocessor
from src.util.constant import Constant

def save(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def train():
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