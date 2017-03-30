%
% Performance evaluation, 2005
% Homework 3, tutorial
%
% Ruben Merz, http://icapeople.epfl.ch/rmerz
%
% Matlab script that performs several iterations of the simpleServer
%

% Clear and clean everything
clear all; close all;

% General parameters setting
maxReq = 1000;
maxLoop = 3;
timeStep = 1000;
tresh = 2000;
minL = Inf;

aggregateTime = [];
aggregateQueueSize = [];

% Performs maxLoop iteration of the simpleServer
for loop=1:1:maxLoop

  fprintf('=> loop: %d ',loop);

  % Perform one simulation of the simple server
  % the variable stat is a matlab structure
  % Do a 'help struct' for more information
  stat = simpleServer(maxReq,timeStep,tresh);

  % Now from the stat struct, let's compute the final statistics
  meanQueueLength(loop,:) = stat.queueLengthCtr/stat.eventTime(end);
  meanResponseTime(loop,:) = stat.responseTimeCtr/stat.request(end);
  meanQueueLengthTresh(loop,:) = stat.queueLengthCtrTresh/(stat.eventTime(end)-tresh);
  meanResponseTimeTresh(loop,:) = stat.responseTimeCtrTresh/(stat.request(end)-stat.request(stat.treshIdx));

  fprintf('meanQueueLength = %f, meanResponseTime = %f\n', meanQueueLength(loop), meanResponseTime(loop));

  % Process the queue size statistics
  if length(stat.sampledIdx) < minL % To take the minimum among all iterations
    minL = length(stat.sampledIdx);
  end
  idx = stat.sampledIdx(1:minL);
  aggregateTime(loop,1:minL) = stat.eventTime(idx);
  aggregateQueueSize(loop,1:minL) = stat.queueSize(idx);
  
end

% Compute the mean of the queue size and of the time
meanQueueSize = mean(aggregateQueueSize(:,1:minL));
meanTime = round(mean(aggregateTime(:,1:minL)));

% Plot the result
plot(meanTime,meanQueueSize);
axis tight; grid on;
xlabel('Time');
ylabel('Mean Queue Length');
% Print the figure to a .ps file
%print -f1 -r600 -depsc2 simpleServer.eps;
