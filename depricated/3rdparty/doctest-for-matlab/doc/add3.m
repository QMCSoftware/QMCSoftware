function sum = add3(value)
% adds 3 to a number
%
% add3(value)
%    returns (value + 3)
%
% Examples:
%
% >> add3(7)
% 
% ans =
% 
%     10
% 
% >> add3([2 4])
% 
% ans =
% 
%      5     7
% 
% >> add3('hi')
% ??? Error using ==> add3 ***
% add3(value) requires value to be a number
% 
%
% TWO blank lines before the prose description of the function continues
%


if ~ isnumeric(value)
    error('add3(value) requires value to be a number');
end

sum = value + 3;