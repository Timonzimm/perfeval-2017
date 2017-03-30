%
% Performance evaluation, 2005
% Homework 3, tutorial
%
% Ruben Merz, http://icapeople.epfl.ch/rmerz
%
% Add an event to the event list of the scheduler
%

function evSched = delEvent(evSched)

% Remove the event at the top of the queue
% We do not have to sort the list, already sorted
evSched.evList = evSched.evList(2:end,:);
evSched.evListLength = evSched.evListLength-1;

%fprintf('Event deleted, evList length = %d\n',evSched.evListLength);
