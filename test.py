import pickle
import os

from constant import Constant

if __name__ == '__main__':
    filename = os.path.join(Constant.MODEL_PATH, 'baseline.pkl')
    model = pickle.load(open(filename, 'rb'))

    while True:
        in_str = input("Put Id: ")
        try:
            recommendation = model.recommend(in_str)
        
            for el in recommendation:
                print(f'Class {el["class"]}')
                print(f'Chapter {el["chapter"]}')
                print(f'{el["eng"]}')
                print()
        except:
            print('No ID Available')