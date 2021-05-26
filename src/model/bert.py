# import warnings
# warnings.filterwarnings('ignore')
# import os

# import bert

# import tensorflow as tf
# import tensorflow_hub as hub
# from tensorflow.keras import Model
# from tensorflow.keras.optimizers import Adam

# import tensorflow_addons as tfa

# from src.util.constant import Constant
# from src.model.callbacks import Checkpoint

# class BertModel(Model):

#     def __init__(self, 
#                  total_words, 
#                  input_length,
#                  n_class):
#         super(BertModel, self).__init__()

#         self.bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)

#     def call(self, inputs, feature_only=False):
#         return self.classificator(x)