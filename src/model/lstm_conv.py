import warnings
warnings.filterwarnings('ignore')
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Dropout, Conv1D, GlobalAveragePooling1D, Concatenate

class ConvLSTMModel(Model):

    def __init__(self, 
                 total_words, 
                 input_length,
                 n_class):
        super(ConvLSTMModel, self).__init__()
        self.dropout = Dropout(rate=0.2)
        self.input_layer = Input((input_length))
        self.embedding = Embedding(total_words, 512, input_length=input_length)
        
        # LSTM
        self.lstm1 = LSTM(256, return_sequences=True)
        self.lstm2 = LSTM(512)
        
        # Conv
        self.conv1 = Conv1D(256, 5, activation='relu')
        self.conv2 = Conv1D(128, 5, activation='relu')
        self.pool1 = GlobalAveragePooling1D()

        # Concatenate
        self.concatenate = Concatenate()

        # Classificator
        self.dense1 = Dense(256, activation='relu')
        self.classificator = Dense(n_class, activation='softmax')
        self.out = self.call(self.input_layer)

    def call(self, inputs, feature_only=False, training=None):
        x = self.embedding(inputs)
        lstm_x = self.lstm1(x)
        lstm_x = self.dropout(lstm_x, training=training)
        lstm_x = self.lstm2(lstm_x)
        lstm_x = self.dropout(lstm_x, training=training)

        conv_x = self.conv1(x)
        conv_x = self.dropout(conv_x, training=training)
        conv_x = self.conv2(conv_x)
        conv_x = self.dropout(conv_x, training=training)
        conv_x = self.pool1(conv_x)

        x = self.concatenate([lstm_x, conv_x])

        if feature_only:
            return x

        x = self.dense1(x)
        return self.classificator(x)