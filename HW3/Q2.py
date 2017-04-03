import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from simulator import Simulator

# QUESTION 2

fig, axes = plt.subplots(nrows=3, ncols=3)

for i, lambda_ in enumerate(range(105, 114, 1)):
  sim = Simulator(lambda_=lambda_)
  results = sim.simulate(num_run=1)
  _, types_deltas_inq = results[0]

  df = pd.DataFrame(types_deltas_inq, columns=['time', 'change all', 'change type 1', 'change type 2'])
  df = df.sort_values(by='time')

  df_final = pd.DataFrame()
  df_final['Time (ms)'] = df['time']
  df_final['Total number of process in queue'] = df['change all'].cumsum(axis=0)
  df_final[df_final < 0] = 0

  print(int(i/3), int(i%3))
  df_final.iloc[::1000, :].plot(x='Time (ms)', ax=axes[int(i/3), int(i%3)])
  axes[int(i/3), int(i%3)].set_title("λ = {} req/s".format(lambda_))

plt.show()