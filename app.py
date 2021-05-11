import pickle
import os
import argparse

from src.util.constant import Constant
from src.test import program
from src.train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--train', action='store_true', required=False, help='Training Model')
    parser.add_argument('-te', '--test', action='store_true', required=False, help='Testing Model')
    parser.add_argument('-v', '--validate', action='store_true', required=False, help='Validate Model')
    args = parser.parse_args()

    if args.train:
        train()
    elif args.test:
        filename = os.path.join(Constant.MODEL_PATH, 'baseline.pkl')
        model = pickle.load(open(filename, 'rb'))

        while True:
            program(model)
    else:
        print('Specify train or test using --train or --test')