import os
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

def load_data(filename, seq_len, normalise_window):
  f = open(filename, 'rb').read()
  data = f.decode().split('\n')

  sequence_length = seq_len + 1
  result = []
  for index in range(len(data) - sequence_length):
      result.append(data[index: index + sequence_length])

  result = np.array(result)

  row = round(0.9 * result.shape[0])
  train = result[:int(row), :]
  np.random.shuffle(train)
  x_train = train[:, :-1]
  y_train = train[:, -1]
  x_test = result[int(row):, :-1]
  y_test = result[int(row):, -1]

  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

  return [x_train, y_train, x_test, y_test]

def build_model(layers):
  model = Sequential()

  model.add(LSTM(
      input_dim=layers[0],
      output_dim=layers[1],
      return_sequences=True))
  model.add(Dropout(0.2))

  model.add(LSTM(
      layers[2],
      return_sequences=False))
  model.add(Dropout(0.2))

  model.add(Dense(
      output_dim=layers[3]))
  model.add(Activation("linear"))

  start = time.time()
  model.compile(loss="mse", optimizer="rmsprop")
  print("> Compilation Time : ", time.time() - start)
  return model