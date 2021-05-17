from src.recommender import Recommender
from src.train import train
from src.test import program_baseline
from src.util.constant import Constant
import argparse
import pandas as pd
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--train', action='store_true',
                        required=False, help='Training Model')
    parser.add_argument('-btr', '--baseline_train', action='store_true',
                        required=False, help='Train Baseline Model')
    parser.add_argument('-te', '--test', action='store_true',
                        required=False, help='Testing Model')
    parser.add_argument('-v', '--validate', action='store_true',
                        required=False, help='Validate Model')
    parser.add_argument('-b', '--baseline', action='store_true',
                        required=False, help='Baseline Recommender')
    parser.add_argument('-r', '--recommender', action='store_true',
                        required=False, help='Recommender Program')
    args = parser.parse_args()

    if args.train:
        train()

    elif args.baseline_train:
        train(baseline=True)

    elif args.baseline:
        filename = os.path.join(Constant.MODEL_PATH, 'baseline.pkl')
        model = pickle.load(open(filename, 'rb'))

        while True:
            program_baseline(model)

    elif args.recommender:
        data = pd.read_csv(os.path.join(
            Constant.DATA_PATH, 'processed', 'train.csv'))
        preprocessor_path = os.path.join(
            Constant.PREPROCESSOR_PATH, 'preprocessor.pkl')
        model_path = os.path.join(Constant.MODEL_PATH, 'convolution',
                                  'Temp (acc_0.03-val_acc_0.03-train_f1_0.00-val_f1_0.00).h5')

        recommender = Recommender(data, preprocessor_path, model_path)
        recommender.recommend()

    else:
        print('Specify train, baseline, or recommender using --train, --baseline, or --recommender')
