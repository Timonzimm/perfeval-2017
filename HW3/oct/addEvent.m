%
% Performance evaluation, 2005
% Homework 3, tutorial
%
% Ruben Merz, http://icapeople.epfl.ch/rmerz
%
%
% Add an event to the event list of the scheduler
%

function evSched = addEvent(eventType,firingTime,evSched)

% Add the event
evSched.evList = [evSched.evList; 
		  firingTime, eventType];
evSched.evListLength = evSched.evListLength + 1;
% Sort the list
evSched.evList = sortrows(evSched.evList);
