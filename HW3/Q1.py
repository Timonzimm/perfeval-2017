import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from simulator import Simulator

sim = Simulator(lambda_=108)
results = sim.simulate(num_run=1)

done_requests, types_deltas_inq = results[0]

# QUESTION 1
# POINT 1
arrivals_type_1 = [Request.arrival_time[0] for Request in done_requests]
departures_type_2 = [Request.service_time[1] + Request.service_duration[1] for Request in done_requests]

plt.plot(arrivals_type_1, list(range(1, len(done_requests) + 1)))
plt.plot(departures_type_2, list(range(len(done_requests), 0, -1)))

plt.show()
# POINT 2
df = pd.DataFrame(types_deltas_inq, columns=['time', 'change all', 'change type 1', 'change type 2'])
df = df.sort_values(by='time')

df_final = pd.DataFrame()
df_final['Time'] = df['time']
df_final['Total number of process in queue'] = df['change all'].cumsum(axis=0)
df_final['Number of type 1 process in queue'] = df['change type 1'].cumsum(axis=0)
df_final['Number of type 2 process in queue'] = df['change type 2'].cumsum(axis=0)
df_final.plot(x='Time')
plt.show()


sim = Simulator(lambda_=70)
results = sim.simulate(num_run=1)

done_requests, types_deltas_inq = results[0]
# POINT 1 bis
mean_response_time_type_1 = np.mean([Request.arrival_time[1] - Request.arrival_time[0] for Request in done_requests])
mean_response_time_type_2 = np.mean([Request.service_time[1] + Request.service_duration[1] - Request.arrival_time[1] for Request in done_requests])

print("Average response time (waiting + service time) for type 1 job: {} ms".format(mean_response_time_type_1))
print("Average response time (waiting + service time) for type 2 job: {} ms".format(mean_response_time_type_2))

# POINT 2 bis