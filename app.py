import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import argparse

from src.util.constant import Constant
from src.test import program_baseline
from src.train import train
from src.recommender import Recommender
from src.process.cleaner import Cleaner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ctr', '--conv_train', action='store_true', required=False, help='Training Convolution Model')
    parser.add_argument('-lstr', '--lstm_train', action='store_true', required=False, help='Training LSTM Model')
    parser.add_argument('-clstr', '--conv_lstm_train', action='store_true', required=False, help='Training ConvLSTM Model')
    parser.add_argument('-be', '--bert', action='store_true', required=False, help='Training Bert Model')
    parser.add_argument('-btr', '--baseline_train', action='store_true', required=False, help='Train Baseline Model')
    parser.add_argument('-te', '--test', action='store_true', required=False, help='Testing Model')
    parser.add_argument('-v', '--validate', action='store_true', required=False, help='Validate Model')
    parser.add_argument('-b', '--baseline', action='store_true', required=False, help='Baseline Recommender')
    parser.add_argument('-r', '--recommender', action='store_true', required=False, help='Recommender Program')
    parser.add_argument('-sr', '--skip_clean', action='store_true', required=False, help='Skip Cleaning Data')
    args = parser.parse_args()

    skip_clean = args.skip_clean
    if args.conv_train:
        train(convolution=True, skip_clean=skip_clean)
    
    elif args.lstm_train:
        train(lstm=True, skip_clean=skip_clean)

    elif args.conv_lstm_train:
        train(conv_lstm=True, skip_clean=skip_clean)

    elif args.baseline_train:
        train(baseline=True, skip_clean=skip_clean)

    elif args.baseline:
        filename = os.path.join(Constant.MODEL_PATH, 'baseline.pkl')
        model = pickle.load(open(filename, 'rb'))
        
        while True:
            program_baseline(model)

    elif args.recommender:
        data = pd.read_csv(os.path.join(Constant.DATA_PATH, 'raw', 'train.csv'))[6213:6215]

        if os.path.isfile(Constant.FEATURES_PATH) and os.path.isfile(Constant.DICT_FEAT_PATH):
            print('[LOG] Found Dumped File')
        else:
            print('[LOG] Cleaning Data')
            cleaner = Cleaner()
            data = cleaner.clean(data)
        
        preprocessor_path = os.path.join(Constant.PREPROCESSOR_PATH, 'preprocessor.pkl')
        model_path = os.path.join(Constant.MODEL_PATH, 'conv_lstm', 'Conv1D-LSTM (acc_0.85-val_acc_0.61-train_f1_0.79-val_f1_0.47).h5')

        recommender = Recommender(data, preprocessor_path, model_path)
        recommender.recommend()

    else:
        print('Specify train, baseline, or recommender using --train, --baseline, or --recommender')