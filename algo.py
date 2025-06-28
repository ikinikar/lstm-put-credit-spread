# region imports
from AlgorithmImports import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras
# endregion

class LSTM_Algo(QCAlgorithm):
    def __init__(self):
        #Hyperparams
        self.lr = 0.001
        self.dropout_size = 0.3
        self.batch_size = 16
        self.epochs = 50
        self.loss = 'mean_squared_error'
        self.metrics = ['accuracy']
        self.units_per_lstm_layer = 50
        self.dense_activation = 'sigmoid'

    def LSTM_model(self, input_shape):
        model = Sequential()
        model.add(Bidirectional(LSTM(self.units_per_lstm_layer, return_sequences=True, input_shape=(input_shape[0], input_shape[1]))))
        model.add(Dropout(self.dropout_size))
        model.add(LSTM(self.units_per_lstm_layer, return_sequences=False))
        model.add(Dropout(self.dropout_size))
        model.add(Dense(1, activation=self.dense_activation))

        optimizer = Adam(learning_rate=self.lr)
        model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)

        return model
