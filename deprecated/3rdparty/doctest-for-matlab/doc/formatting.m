% formatting examples
%
% >> 1 + 1          % should work fine
% 
% ans =
% 
%      2
%
% >> 1 + 1          % comparisons collapse all whitespace, so this passes
% ans = 2
% 
% >> 1 + 1;         % expects no output, since >> is on the next line
% >> for I = 1:3    % FAILS: code to run can only be one line long
% disp(I)
% end
%      1
% 
%      2
% 
%      3
% 
% >> for I = 1:3; disp(I); end      % but this works
%      1
% 
%      2
% 
%      3
% 
% >> 1 + 4          % FAILS: there aren't 2 blank lines before the prose
% 
% ans =
% 
%      5
% 
% Blah blah blah oops!  This prose started too soon!
%
%
% Sometimes you have output that changes each time you run a function
% >> dicomuid       % FAILS: no wildcard on changing output
% 
% ans =
% 
% 1.3.6.1.4.1.9590.100.1.1.944807727511025110.343357080818013
%
%
% You can use *** as a wildcard to match this!
% >> dicomuid       % passes
% 
% ans =
% 
% 1.3.6.1.4.1.***
%
%
% I guess that's it!

