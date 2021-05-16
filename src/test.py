import os
import pickle

from tensorflow.keras.models import load_model

from src.util.constant import Constant

def program_baseline(model):
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