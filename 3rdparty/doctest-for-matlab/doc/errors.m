% Errors and doctest - demonstrates a current limitation of doctest
%
% This one works fine.
%
% >> not_a_real_function(42)
% ??? Undefined function or method 'not_a_real_function' for input
% arguments of type 'double'.
%
%
% This one breaks.
%
% >> disp('if at first you don''t succeed...'); error('nevermind')
% if at first you don't succeed...
% ??? nevermind
