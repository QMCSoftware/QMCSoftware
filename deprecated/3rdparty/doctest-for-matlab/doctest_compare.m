function match = doctest_compare(want, got)
% Matches two strings together... they should be identical, except that the
% first one can contain '***', which matches anything in the second string.
%
% But there are also some tricksy things that Matlab does to strings.  Such
% as add hyperlinks to help.  This doctest tests that condition.
%
% >> disp('Hi there!  <a href="matlab:help help">foo</a>')
% Hi there!  foo
%
%
% They also sometimes backspace over things for no apparent reason.  This
% doctest recreates that condition.
%
% >> sprintf('Hi there, there is no dot here: .\x08')
% 
% ans =
% 
% Hi there, there is no dot here: 
%
%
% All of the doctests should pass, and they manipulate this function.
%

% This looks bad, like hardcoding for lower-case "a href"
% and a double quote... but that's what MATLAB looks for too.
got = regexprep(got, '<a +href=".*?>', '');
got = regexprep(got, '</a>', '');


% WHY do they need backspaces?  huh.
got = regexprep(got, '.\x08', '');

want = strtrim(want);
got = strtrim(got);
got =  regexprep(got, '×', 'x');

if isempty(want) && isempty(got)
    match = 1;
    return
end

want_escaped = regexptranslate('escape', want);
want_re = regexprep(want_escaped, '(\\\*){3}', '.*');



result = regexp(got, want_re, 'once');

match = ~ isempty(result);

end