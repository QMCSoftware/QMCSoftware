%% DOCTEST - Run examples embedded in documentation
%
% With doctest, you can put an example of using your function, right in the
% m-file help.  Then, that same example can be used like a unit test, to
% make sure the function still does what the docs say it does.
%
% Here's a trivial function and its documentation:
%

type add3

%% Example output
%
% Now we'll run
%
% doctest add3
%
% Here's the output we get:
%

doctest add3



%% Failure
% Here's an example of what happens when something changes and your test
% fails.
%
% By the way, output is in the Test Anything Protocol format, which I guess
% is mostly used by Perl people, but it's good enough for now.  See 
% http://testanything.org/
%
% Normally, the failure report would include a link to somewhere near the
% doctest that failed, but that doesn't format properly in published
% m-files.
%

type should_fail
disp -------------
doctest('should_fail', 'CreateLinks', 0) % the links don't work in publish()



%% Defining your expectations
%
% Each time doctest runs a test, it's running a line of code and checking
% that the output is what you say it should be.  It knows something is an
% example because it's a line in help('your_function') that starts with
% '>>'.  It knows what you think the output should be by starting on the
% line after >> and looking for the next >>, two blank lines, or the end of
% the documentation.
%
% If the output of some function will change each time you call it, for
% instance if it includes a random number or a stack trace, you can put
% '***' (three asterisks) where the changing element should be.  This acts
% as a wildcard, and will match anything.  See the example below.
%
% Here are some examples of formatting, both ones that work and ones that
% don't.
%

type formatting
disp -------------
doctest('formatting', 'CreateLinks', 0)



%% Expecting an error
%
% doctest can deal with errors, a little bit.  You might want this to test
% that your function correctly detects that it is being given invalid
% parameters.  But if your example will emit other output BEFORE the error
% message, the current version can't deal with that.  For more info see
% Issue #4 on the bitbucket site (below).  Warnings are different from
% errors, and they work fine.

type errors
disp -------------
doctest('errors', 'CreateLinks', 0)



%% Limitations
%
% All adjascent white space is collapsed into a single space before
% comparison, so right now doctest can't detect a failure that's purely a
% whitespace difference.
%
% It can't run examples that are longer than one line of code (so, for
% example, no loops that take more than one line).  This is difficult
% because I haven't found a good way to mark these subsequent lines as
% part-of-the-source-code rather than part-of-the-result.  However,
% variables that you define in one line do carry over to the next.
%
% I haven't found a good way of isolating the variables that you define in
% the tests from the variables used to run the test.  So, don't run CLEAR
% in your doctest, and don't expect WHO/WHOS to work right, and don't mess
% with any variables that start with DOCTEST__.  :-/
% 
% When you're working on writing/debugging a Matlab class, you might need
% to run 'clear classes' to get correct results from doctests (this is a
% general problem with developing classes in Matlab).
%
% The latest version from the original author, Thomas Smith, is available
% at http://bitbucket.org/tgs/doctest-for-matlab/src
%
% The bugtracker is also there, let me know if you encounter any problems!
%
%
