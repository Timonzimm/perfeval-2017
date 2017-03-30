%
% Performance evaluation, 2005
% Homework 3, tutorial
%
% Ruben Merz, http://icapeople.epfl.ch/rmerz
%
% Matlab implementation of a "simple server"
%

function stat = simpleServer(maxReq,timeStep,tresh)

% Initialize the random number generator
rand('state',sum(100*clock));

% Create the eventScheduler
% And bootstrap it (i.e. add the first event)
evSched.currentTime = -1;
evSched.firingTime = 0;
evSched.evList = [,]; % This is the event list, note that it is a [n x 2] vector.
		      % The first column corresponds to the firing time
		      % The second column is the event type
		      % 1 for an arrival
		      % 2 for a service
		      % 3 for a departure
evSched.evListLength = 0; % And a variable corresponding to the event list size (i.e number of rows)
evSched = addEvent(1,0,evSched); % Add first event i.e. 1 for arrival with firing Time = 0

% Initialize buffer queues
server.queue = [];
server.queueLength = 0;
server.nbArrival = 0;
server.nbRequest = 0;

% Variables used for statistic purpose, refer to the perfeval lecture notes for the terminology (simulation chapter)
stat.eventTime = [];
stat.treshIdx = 0;
stat.sampledIdx = 1;
stat.queueSize = [];
stat.request = [];
stat.queueLengthCtr = 0;
stat.queueLengthCtrTresh = 0;
stat.responseTimeCtr = 0;
stat.responseTimeCtrTresh = 0;

% Execute Events until the total number of arrivals reach maxReq
while server.nbArrival <= maxReq

  % Get next event
  evType = evSched.evList(1,2); % Assumes that evList is sorted
  evSched.firingTime = evSched.evList(1,1);

  % Remove it from the event list
  evSched = delEvent(evSched);

  % Increment counter for the arrival queue
  stat.queueLengthCtr = ...
      stat.queueLengthCtr + server.queueLength*(evSched.firingTime-evSched.currentTime);
  if evSched.firingTime > tresh
    stat.queueLengthCtrTresh = ...
	stat.queueLengthCtrTresh + server.queueLength*(evSched.firingTime-evSched.currentTime);
  end
  % Register queue size
  stat.eventTime(end+1) = evSched.firingTime;
  stat.queueSize(end+1) = server.queueLength;
  stat.request(end+1) = server.nbRequest;  
  
  if mod(evSched.firingTime,timeStep) < mod(evSched.currentTime,timeStep)
    stat.sampledIdx(end+1) = length(stat.eventTime);
  end
  % Execute current event
  % Since the data structures are not shared, the discrimination between
  % events is done here. Then, given the event type, there is a specific
  % execution function to call and a set of parameter to update
  switch evType

   case 1, % Arrival fprintf('=> Arrival Event: evFiringTime = %d\n',evSched.firingTime);

    server.nbArrival = server.nbArrival + 1;

    % Create a new request and add it at the end of the buffer
    server.queue = [server.queue; evSched.firingTime];
    server.queueLength = server.queueLength + 1;

    % Add a service event if the queue was empty
    if server.queueLength == 1
      evSched = addEvent(2,evSched.firingTime,evSched);
    end
    % Draw a random number from distrib F to create a new Arrival event
    delta = getDelta('F');
    evSched = addEvent(1,evSched.firingTime+delta,evSched);
   
   case 2, % Service fprintf('=> Service Event: evFiringTime = %d\n',evSched.firingTime);

    % Draw a random number from distrib G to create a new Departure event
    delta = getDelta('G');
    evSched = addEvent(3,evSched.firingTime+delta,evSched);

   case 3, % Departure fprintf('=> Departure Event: evFiringTime = %d\n',evSched.firingTime);

    % Update response time counters
    stat.responseTimeCtr = stat.responseTimeCtr + evSched.firingTime-server.queue(1);
    if evSched.firingTime > tresh
      stat.responseTimeCtrTresh = stat.responseTimeCtrTresh + evSched.firingTime-server.queue(1);
    end
    % Job has been served and  we don't need its arrival time anymore
    server.nbRequest = server.nbRequest + 1;

    % Remove request from buffer and delete it
    server.queue = server.queue(2:end);
    server.queueLength = server.queueLength - 1;
 
    % Add a service event if the queue is not empty
    if server.queueLength > 0
      evSched = addEvent(2,evSched.firingTime,evSched);
    end

   otherwise
    fprintf('Bug: this type of event does not exist!\n');

  end

  % Set current time to ev.firingTime
  evSched.currentTime = evSched.firingTime;
  if evSched.currentTime > tresh && stat.treshIdx == 0;
    stat.treshIdx = length(stat.eventTime);
  end

end
