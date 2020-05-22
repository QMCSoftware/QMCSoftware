function results = doctest_run(docstring)
%DOCTEST_RUN - used internally by doctest
%
% Usage:
%   doctest_run(docstring)
%       Runs all the examples in the given docstring and returns a
%       structure with the results from running.
%
% The return value is a structure with the following fields:
%
% results.source:   the source code that was run
% results.want:     the desired output
% results.got:      the output that was recieved
% results.pass:     whether .want and .got match each other according to
%       doctest_compare.
%

% loosely based on Python 2.6 doctest.py, line 510
example_re = '(?m)(?-s)(?:^ *>> )(?<source>.*)\n(?<want>(?:(?:^ *$\n)?(?!\s*>>).*\w.*\n)*)';

[examples] = regexp(docstring, example_re, 'names', 'warnings');

results = [];

all_outputs = DOCTEST__evalc({examples(:).source});
  
for I = 1:length(examples)
  
    got = all_outputs{I};
    want_unspaced = regexprep(examples(I).want, '\s+', ' ');
    
    got_unspaced = regexprep(got, '\s+', ' ');
    

    
    results(I).source = examples(I).source;
    results(I).want = strtrim(want_unspaced);
    results(I).got = strtrim(got_unspaced);
    results(I).pass = doctest_compare(want_unspaced, got_unspaced);
    
end

end



function DOCTEST__results = DOCTEST__evalc(DOCTEST__examples_to_run)
% I wish I had my very own namespace...
% Structure adapted from a StackOverflow answer by user Amro:
% http://stackoverflow.com/questions/3283586
% http://stackoverflow.com/users/97160/amro

DOCTEST__results = cell(size(DOCTEST__examples_to_run));

for DOCTEST__I = 1:numel(DOCTEST__examples_to_run)
    try
        DOCTEST__results{DOCTEST__I} = evalc(DOCTEST__examples_to_run{DOCTEST__I});
    catch DOCTEST__exception
        DOCTEST__results{DOCTEST__I} = DOCTEST__format_exception(DOCTEST__exception);
    end
end



% If we get excited, we could add this snippet by Amro
%             % list created variables in this context
%             %clear ans
%             DOCTEST__vars = whos('-regexp', '^(?!DOCTEST__).*');   % java regex negative lookahead
%             varargout{1} = { DOCTEST__vars.name };
% 
%             if nargout > 2
%                 % return those variables
%                 varargout{2} = cell(1,numel(DOCTEST__vars));
%                 for DOCTEST__i=1:numel(DOCTEST__vars)
%                     [~,varargout{2}{DOCTEST__i}] = evalc( DOCTEST__vars(DOCTEST__i).name );
%                 end
%             end

end

function formatted = DOCTEST__format_exception(ex)

if strcmp(ex.stack(1).name, 'DOCTEST__evalc')
    % we don't want the report, we just want the message
    % otherwise it'll talk about evalc, which is not what the user got on
    % the command line.
    formatted = ['??? ' ex.message];
else
    formatted = ['??? ' ex.getReport('basic')];
end



end



