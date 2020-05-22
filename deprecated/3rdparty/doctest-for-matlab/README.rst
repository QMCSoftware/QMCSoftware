README
Contents
--------


-  `DOCTEST - Run examples embedded in documentation <#id1>`_
-  `Example output <#id2>`_
-  `Failure <#id3>`_
-  `Defining your expectations <#id4>`_
-  `Expecting an error <#id5>`_
-  `Limitations <#id6>`_

DOCTEST - Run examples embedded in documentation
------------------------------------------------

With doctest, you can put an example of using your function, right
in the m-file help. Then, that same example can be used like a unit
test, to make sure the function still does what the docs say it
does.

Here's a trivial function and its documentation:

::

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

Example output
--------------

Now we'll run

doctest add3

Here's the output we get:

::

    TAP version 13
    1..3
    ok 1 - "add3(7)"
    ok 2 - "add3([2 4])"
    ok 3 - "add3('hi')"

Failure
-------

Here's an example of what happens when something changes and your
test fails.

By the way, output is in the Test Anything Protocol format, which I
guess is mostly used by Perl people, but it's good enough for now.
See `http://testanything.org/ <http://testanything.org/>`_

Normally, the failure report would include a link to somewhere near
the doctest that failed, but that doesn't format properly in
published m-files.

::

    % Has a doctest that should fail.
    %
    % >> 3 + 3
    % 
    % ans =
    %
    %      5
    %
    
    -------------
    TAP version 13
    1..1
    not ok 1 - "3 + 3"
        expected: ans = 5
        got     : ans = 6

Defining your expectations
--------------------------

Each time doctest runs a test, it's running a line of code and
checking that the output is what you say it should be. It knows
something is an example because it's a line in
help('your\_function') that starts with '>>'. It knows what you
think the output should be by starting on the line after >> and
looking for the next >>, two blank lines, or the end of the
documentation.

If the output of some function will change each time you call it,
for instance if it includes a random number or a stack trace, you
can put '\*\*\*' (three asterisks) where the changing element
should be. This acts as a wildcard, and will match anything. See
the example below.

Here are some examples of formatting, both ones that work and ones
that don't.

::

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
    
    
    -------------
    TAP version 13
    1..8
    ok 1 - "1 + 1          % should work fine"
    ok 2 - "1 + 1          % comparisons collapse all whitespace, so this passes"
    ok 3 - "1 + 1;         % expects no output, since >> is on the next line"
    not ok 4 - "for I = 1:3    % FAILS: code to run can only be one line long"
        expected: disp(I) end 1 2 3
        got     : ??? Error: At least one END is missing: the statement may begin here.
    ok 5 - "for I = 1:3; disp(I); end      % but this works"
    not ok 6 - "1 + 4          % FAILS: there aren't 2 blank lines before the prose"
        expected: ans = 5 Blah blah blah oops! This prose started too soon!
        got     : ans = 5
    not ok 7 - "dicomuid       % FAILS: no wildcard on changing output"
        expected: ans = 1.3.6.1.4.1.9590.100.1.1.944807727511025110.343357080818013
        got     : ans = 1.3.6.1.4.1.9590.100.1.2.282084865731251048027101806490961582150
    ok 8 - "dicomuid       % passes"

Expecting an error
------------------

doctest can deal with errors, a little bit. You might want this to
test that your function correctly detects that it is being given
invalid parameters. But if your example will emit other output
BEFORE the error message, the current version can't deal with that.
For more info see Issue #4 on the bitbucket site (below). Warnings
are different from errors, and they work fine.

::

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
    
    -------------
    TAP version 13
    1..2
    ok 1 - "not_a_real_function(42)"
    not ok 2 - "disp('if at first you don''t succeed...'); error('nevermind')"
        expected: if at first you don't succeed... ??? nevermind
        got     : ??? nevermind

Limitations
-----------

All adjascent white space is collapsed into a single space before
comparison, so right now doctest can't detect a failure that's
purely a whitespace difference.

It can't run examples that are longer than one line of code (so,
for example, no loops that take more than one line). This is
difficult because I haven't found a good way to mark these
subsequent lines as part-of-the-source-code rather than
part-of-the-result. However, variables that you define in one line
do carry over to the next.

I haven't found a good way of isolating the variables that you
define in the tests from the variables used to run the test. So,
don't run CLEAR in your doctest, and don't expect WHO/WHOS to work
right, and don't mess with any variables that start with
DOCTEST\_\_. :-/

When you're working on writing/debugging a Matlab class, you might
need to run 'clear classes' to get correct results from doctests
(this is a general problem with developing classes in Matlab).

The latest version from the original author, Thomas Smith, is
available at
`http://bitbucket.org/tgs/doctest-for-matlab/src <http://bitbucket.org/tgs/doctest-for-matlab/src>`_

The bugtracker is also there, let me know if you encounter any
problems!

Published with MATLAB® 7.10


