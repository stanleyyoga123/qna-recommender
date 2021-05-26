import warnings
warnings.filterwarnings('ignore')
import os

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam

class ConvModel(Model):

    def __init__(self, 
                 total_words, 
                 input_length,
                 n_class):
        super(ConvModel, self).__init__()
        self.input_layer = Input((input_length))
        self.embedding = Embedding(total_words, 512, input_length=input_length)
        self.conv1 = Conv1D(256, 5, activation='relu')
        self.conv2 = Conv1D(128, 5, activation='relu')
        self.pool1 = GlobalAveragePooling1D()
        self.dense1 = Dense(64, activation='relu')
        self.classificator = Dense(n_class, activation='softmax')
        self.out = self.call(self.input_layer)

    def call(self, inputs, feature_only=False):
        x = self.embedding(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        
        if feature_only:
            return x

        x = self.dense1(x)
        return self.classificator(x)