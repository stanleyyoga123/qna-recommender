from src.util.constant import Constant
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Embedding, Dense, Conv1D, GlobalAveragePooling1D
from tensorflow.keras import Model
import tensorflow as tf
import os
import warnings
warnings.filterwarnings('ignore')


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


class Checkpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super(Checkpoint, self).__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs={}):
        print('Saving Checkpoint')
        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        train_f1 = logs.get('f1_score')
        val_f1 = logs.get('val_f1_score')
        self.model.save_weights(
            f'{self.filepath} (acc_{train_acc:.2f}-val_acc_{val_acc:.2f}-train_f1_{train_f1:.2f}-val_f1_{val_f1:.2f}).h5')


def train_model(x_train,
                y_train,
                x_test,
                y_test,
                total_words,
                input_length,
                n_class,
                epochs=10,
                batch_size=256):

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

    f1 = tfa.metrics.F1Score(n_class, 'macro')

    model = ConvModel(total_words, input_length, n_class)
    model.build((None, x_train.shape[1]))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(), metrics=['accuracy', f1])

    print(model.summary())

    filepath = os.path.join(Constant.MODEL_PATH, 'convolution', 'Temp')
    callbacks = [Checkpoint(filepath)]

    history = model.fit(x_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        callbacks=callbacks,
                        verbose=1)
    return model
