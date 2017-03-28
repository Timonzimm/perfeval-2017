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