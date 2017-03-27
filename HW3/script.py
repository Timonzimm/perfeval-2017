import numpy as np
import pandas as pd
from collections import deque
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

max_arrivals = 10000 # max number of request arrivals (set to 10⁴ from the homework description)
lambda_ = 70000 # Exponential parameter, arrival rate
log_normal = (2, 0.5) # Log-normal parameters mu and sigma, type 1 process service time
uniform = (0.7, 0.9) # Uniform parameters, type 2 process service time

waiting_requests = deque([])
done_requests = deque([])
processes = []

class Request:
  def __init__(self, arrival_time, id, service_duration, type_ = 1):
    self.arrival_time = [arrival_time]
    self.service_time = [-1, -1] # Service start for type 1 -> type 2 and for type 2 -> end
    self.service_duration = service_duration # Service duration for type 1 -> type 2 and for type 2 -> end
    self.type_ = type_
    self.id = id
  def __repr__(self):
    return self.__str__()
  def __str__(self):
    return "\n\n***** {} *****\narrival_time: {}\nservice_time: {}\nservice_duration: {}".format(
      self.id,
      self.arrival_time,
      self.service_time,
      self.service_duration
    )

t = np.random.exponential(scale=1/lambda_)
processes.append((t, +1, +1, 0))
waiting_requests.appendleft(Request(
  arrival_time=t,
  service_duration=[np.random.lognormal(mean=log_normal[0], sigma=log_normal[1]), np.random.uniform(low=uniform[0], high=uniform[1])],
  id=1))

for i in range(1, max_arrivals):
  t = waiting_requests[0].arrival_time[0] + np.random.exponential(scale=1/lambda_)
  processes.append((t, +1, +1, 0))
  waiting_requests.appendleft(Request(
    arrival_time=t,
    service_duration=[np.random.lognormal(mean=log_normal[0], sigma=log_normal[1]), np.random.uniform(low=uniform[0], high=uniform[1])],
    id=i+1))

def get_index(q, to_insert):
  for i, elem in enumerate(q):
    if elem.arrival_time[-1] < to_insert.arrival_time[-1]:
      return i
  
  return len(q)


last_end_service_time = waiting_requests[-1].arrival_time[0]
while len(waiting_requests) > 0:
  processing = waiting_requests.pop()
  #print("Processing request {}, type {}".format(processing.id, processing.type_))

  if processing.type_ == 1:
    processing.service_time[0] = max(last_end_service_time, processing.arrival_time[0])
    processing.type_ = 2

    last_end_service_time = processing.service_time[0] + processing.service_duration[0]

    processing.arrival_time.append(last_end_service_time)
    index = get_index(waiting_requests, processing)

    waiting_requests.insert(index, processing)
    processes.append((processing.service_time[0], 0, -1, +1))

  elif processing.type_ == 2:
    processing.service_time[1] = max(last_end_service_time, processing.arrival_time[1])

    last_end_service_time = processing.service_time[1] + processing.service_duration[1]

    done_requests.appendleft(processing)
    processes.append((processing.service_time[1], -1, 0, -1))

    if processing.arrival_time[1] != processing.service_time[0] + processing.service_duration[0]:
      print("ERROR 1")
      print(processing)
      exit()
    elif processing.arrival_time[1] < processing.arrival_time[0] + processing.service_duration[0]: 
      print("ERROR 2")
      print(processing)
      exit()
  else:
    print("ERROR 3")
    exit()



arrivals_type_1 = [Request.arrival_time[0] for Request in reversed(done_requests)]
departures_type_2 = [Request.service_time[1] + Request.service_duration[1] for Request in reversed(done_requests)]

plt.plot(arrivals_type_1, list(range(1, len(done_requests) + 1)))
plt.plot(departures_type_2, list(range(len(done_requests), 0, -1)))

plt.show()

df = pd.DataFrame(processes, columns=['time', 'change all', 'change type 1', 'change type 2'])
df = df.sort_values(by='time')

df_final = pd.DataFrame()
df_final['Time'] = df['time']
df_final['Number of process in queue'] = df['change all'].cumsum(axis=0)
df_final['Number of type 1 process in queue'] = df['change type 1'].cumsum(axis=0)
df_final['Number of type 2 process in queue'] = df['change type 2'].cumsum(axis=0)
#df_final['verif'] = df_final['Number of process'] - df_final['Number of type 1 process'] - df_final['Number of type 2 process']

#print(df_final)

df_final.plot(x='Time')
plt.show()