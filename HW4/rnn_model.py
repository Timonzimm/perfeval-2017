import keras
from keras.layers.wrappers import TimeDistributed
from keras.layers import Dense, LSTM

import numpy as np
import itertools as it

class Data:
    def __init__(self, path, batch_size, seq_len, train_size):
        self.path = path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.train_size = train_size
        self.raw_data = np.loadtxt(path).T

    def prepare_batch(self, ts):
        def seq_gen():
            xy_len = self.seq_len 
            idx = np.arange(0, ts.size - xy_len - 1)
            for i in idx:
                ts_x = ts[i:i+xy_len]
                ts_y = ts[i+1:i+1+xy_len]
                yield (ts_x, ts_y)
        
        SG = seq_gen()
        while True:
            batch = it.islice(SG, self.batch_size)
            batch = list(batch)
             
            x = np.array([np.reshape(x[0], (self.seq_len, 1)) for x in batch])
            y = np.array([np.reshape(x[1], (self.seq_len, 1)) for x in batch])
            
            if (x.shape != (self.batch_size, self.seq_len, 1)):
                SG = seq_gen()
                continue
            yield (x, y)
            
    def batch_gen(self):
        total_len = self.raw_data.size
        separ = int(self.train_size*total_len)
        training_data = self.raw_data[0:separ]
        validation_data = self.raw_data[separ:-1]
        
        G = self.prepare_batch(training_data) 
        
        return G

batch_size = 10 
seq_len = 100 
units = 128
dp = 0.2

d = Data(path='m.csv', 
         batch_size=batch_size,
         seq_len=seq_len,
         train_size=0.8)
        
G = d.batch_gen()

model = keras.models.Sequential()
model.add(LSTM(units, input_shape=(seq_len, 1), return_sequences=True, dropout=dp))
model.add(LSTM(units, input_shape=(seq_len, 1), return_sequences=True, dropout=dp))
model.add(LSTM(units, input_shape=(seq_len, 1), return_sequences=True, dropout=dp))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mse', optimizer='adam')
filepath='models/lstm/mdl.h5'


epoch_save = keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, log: model.save(filepath, overwrite=True))


callbacks_list = [epoch_save]


model.summary()
model.fit_generator(G, 1000, 10, callbacks=callbacks_list)


