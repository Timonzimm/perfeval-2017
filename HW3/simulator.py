import numpy as np
from collections import deque
from request import Request

class Simulator:
  def __init__(self, max_arrivals=10000, lambda_ = 70, log_normal=(2, 0.5), uniform=(0.7, 0.9)):
    self.max_arrivals = max_arrivals # max number of request arrivals (set to 10⁴ from the homework description)
    self.lambda_ = lambda_ / 1000 # Exponential parameter, arrival rate (109 req/s stat. vs 110 req/s transient)
    self.log_normal = log_normal # Log-normal parameters mu and sigma, type 1 process service time
    self.uniform = uniform # Uniform parameters, type 2 process service time
    self.waiting_requests = deque([])
    self.done_requests = []
    self.types_deltas_inq = []

    # Generating all arrival times
    t = np.random.exponential(scale=1/lambda_)
    self.types_deltas_inq.append((t, +1, +1, 0))
    self.waiting_requests.appendleft(Request(
      arrival_time=t,
      service_duration=[np.random.lognormal(mean=log_normal[0], sigma=log_normal[1]), np.random.uniform(low=uniform[0], high=uniform[1])],
      id=1))

    for i in range(1, max_arrivals):
      t = self.waiting_requests[0].arrival_time[0] + np.random.exponential(scale=1/lambda_)
      self.types_deltas_inq.append((t, +1, +1, 0))
      self.waiting_requests.appendleft(Request(
        arrival_time=t,
        service_duration=[np.random.lognormal(mean=log_normal[0], sigma=log_normal[1]), np.random.uniform(low=uniform[0], high=uniform[1])],
        id=i+1))

  def __reset__(self):
    self.waiting_requests = deque([])
    self.done_requests = []
    self.types_deltas_inq = []

    # Generating all arrival times
    t = np.random.exponential(scale=1/self.lambda_)
    self.types_deltas_inq.append((t, +1, +1, 0))
    self.waiting_requests.appendleft(Request(
      arrival_time=t,
      service_duration=[np.random.lognormal(mean=self.log_normal[0], sigma=self.log_normal[1]), np.random.uniform(low=self.uniform[0], high=self.uniform[1])],
      id=1))

    for i in range(1, self.max_arrivals):
      t = self.waiting_requests[0].arrival_time[0] + np.random.exponential(scale=1/self.lambda_)
      self.types_deltas_inq.append((t, +1, +1, 0))
      self.waiting_requests.appendleft(Request(
        arrival_time=t,
        service_duration=[np.random.lognormal(mean=self.log_normal[0], sigma=self.log_normal[1]), np.random.uniform(low=self.uniform[0], high=self.uniform[1])],
        id=i+1))

  def simulate(self, num_run=1):
    res = []

    def get_index(q, to_insert):
      for i, elem in enumerate(q):
        if elem.arrival_time[-1] < to_insert.arrival_time[-1]:
          return i
      
      return len(q)

    for i in range(num_run):
      self.__reset__()

      last_end_service_time = self.waiting_requests[-1].arrival_time[0]
      while len(self.waiting_requests) > 0:
        processing = self.waiting_requests.pop()
        #print("Processing request {}, type {}".format(processing.id, processing.type_))

        if processing.type_ == 1:
          processing.service_time[0] = max(last_end_service_time, processing.arrival_time[0])
          processing.type_ = 2

          last_end_service_time = processing.service_time[0] + processing.service_duration[0]

          processing.arrival_time.append(last_end_service_time)
          index = get_index(self.waiting_requests, processing)

          self.waiting_requests.insert(index, processing)

          self.types_deltas_inq.append((processing.service_time[0], -1, -1, 0))
          self.types_deltas_inq.append((last_end_service_time, +1, 0, +1))

        elif processing.type_ == 2:
          processing.service_time[1] = max(last_end_service_time, processing.arrival_time[1])

          last_end_service_time = processing.service_time[1] + processing.service_duration[1]

          self.done_requests.append(processing)

          self.types_deltas_inq.append((processing.service_time[1], -1, 0, -1))
          self.types_deltas_inq.append((last_end_service_time, 0, 0, 0))
      
      res.append((self.done_requests, self.types_deltas_inq))

    return res