import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from simulator import Simulator

# QUESTION 2

fig, axes = plt.subplots(nrows=3, ncols=3)

for i, lambda_ in enumerate(range(50, 251, 25)):
  sim = Simulator(lambda_=lambda_)
  results = sim.simulate(num_run=1)
  _, types_deltas_inq = results[0]

  df = pd.DataFrame(types_deltas_inq, columns=['time', 'change all', 'change type 1', 'change type 2'])
  df = df.sort_values(by='time')

  df_final = pd.DataFrame()
  df_final['Time'] = df['time']
  df_final['Total number of process in queue'] = df['change all'].cumsum(axis=0)

  df_final.plot(x='Time', ax=axes[int(i/2), int(i%3)])
  axes[int(i/2), int(i%3)].set_title("λ = {}".format(lambda_))

axes[2,1].axis('off')
plt.show()