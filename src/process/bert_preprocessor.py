import bert
import tensorflow_hub as hub
import tensorflow as tf

import numpy as np

from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

class BertPreprocessor:

    def __init__(self):
        FullTokenizer = bert.bert_tokenization.FullTokenizer
        bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)
        vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
        
        self.maxlen = 512
        self.n_class = 0
        self.tokenizer = FullTokenizer(vocab_file, do_lower_case)
        self.encoder = OneHotEncoder()

    def encode_sentence(self, eng):
        return ["[CLS]"] + self.tokenizer.tokenize(eng) + ["[SEP]"]

    def get_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def get_mask(self, tokens):
        return np.char.not_equal(tokens, "[PAD]").astype(int)

    def get_segments(self, tokens):
        seg_ids = []
        current_seg_id = 0
        for tok in tokens:
            seg_ids.append(current_seg_id)
            if tok == "[SEP]":
                current_seg_id = 1-current_seg_id
        return seg_ids

    def padding(encoded_sentence, length=512):
        length_now = len(encoded_sentence)
        if length_now > length:
            ret = encoded_sentence[:511] + ['[SEP]']
            return ret
        pad = ['[PAD]' for _ in range(length-length_now)]
        ret = encoded_sentence + pad
        return ret
    
    def fit(self, data):
        self.encoder.fit(data[['chapter']])

    def transform(self, data, meta=False):
        labels = self.encoder.transform(data[['chapter']]).toarray()

        print('Encode & Pad sentence')
        encoded = []
        for eng in tqdm(data['eng'], total=len(data['eng'])):
            encoded.append(self.padding(self.encode_sentence(eng)))

        print('Convert to Bert Input')
        bert_input = []
        for el in tqdm(encoded, total=len(encoded)):
            bert_input.append(
                tf.stack(
                    [tf.cast(self.get_ids(el), dtype=tf.int32),
                     tf.cast(self.get_mask(el), dtype=tf.int32),
                     tf.cast(self.get_segments(el), dtype=tf.int32)],
                    axis=0 
                )
            )
        bert_input = tf.stack(bert_input)

        if not meta:
            self.n_class = len(labels[0])

        return bert_input, labels

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

        