import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

duration = 100 # duration of the simulation
lambda_ = 10 # Exponential parameter, arrival rate
log_normal = (2, 0.5) # Log-normal parameters mu and sigma
uniform = (0.7, 0.9) # Uniform parameters

processes = []

t = 0
while t <= duration:
  t += np.random.exponential(scale=1/lambda_)
  end_time_1 = t + np.random.lognormal(mean=log_normal[0], sigma=log_normal[1])
  end_time_2 = end_time_1 + np.random.uniform(low=uniform[0], high=uniform[1])

  processes.append((t, +1, +1, 0))
  processes.append((end_time_1, 0, -1, +1))
  processes.append((end_time_2, -1, 0, -1))

df = pd.DataFrame(processes, columns=['time', 'change all', 'change type 1', 'change type 2'])
df = df.sort_values(by='time')

df_final = pd.DataFrame()
df_final['Time'] = df['time']
df_final['Number of process'] = df['change all'].cumsum(axis=0)
df_final['Number of type 1 process'] = df['change type 1'].cumsum(axis=0)
df_final['Number of type 2 process'] = df['change type 2'].cumsum(axis=0)
#df_final['verif'] = df_final['Number of process'] - df_final['Number of type 1 process'] - df_final['Number of type 2 process']

print(df_final)

df_final.plot(x='Time')
plt.show()
