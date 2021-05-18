import warnings
warnings.filterwarnings('ignore')
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam

import tensorflow_addons as tfa

from src.util.constant import Constant
from src.model.callbacks import Checkpoint

class LSTMModel(Model):

    def __init__(self, 
                 total_words, 
                 input_length,
                 n_class):
        super(LSTMModel, self).__init__()
        self.input_layer = Input((input_length))
        self.embedding = Embedding(total_words, 512, input_length=input_length)
        self.bilstm1 = LSTM(64, return_sequences=True)
        self.bilstm2 = LSTM(128)
        self.dense1 = Dense(256, activation='relu')
        self.classificator = Dense(n_class, activation='softmax')
        self.out = self.call(self.input_layer)

    def call(self, inputs, feature_only=False):
        x = self.embedding(inputs)
        x = self.bilstm1(x)
        x = self.bilstm2(x)
        
        if feature_only:
            return x

        x = self.dense1(x)
        return self.classificator(x)