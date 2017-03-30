%
% Performance evaluation, 2005
% Homework 3, tutorial
%
% Ruben Merz, http://icapeople.epfl.ch/rmerz
%
%
%

function delta = getDelta(type)

lambdaF = 0.0095;
lambdaG = 0.01;

switch type
 case 'F'
  delta = exprnd(1/lambdaF);
 case 'G'
  delta = exprnd(1/lambdaG);
end
