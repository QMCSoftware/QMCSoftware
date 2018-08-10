% To generate .../doctest/README.html, 
% change directory to .../doctest/doc
% then run gen_readme
% 
% Then in the shell run html2markdown.py README.html > README.markdown
%

opts = [];
opts.format = 'html';
opts.outputDir = '..';
opts.showCode = false;

publish('README', opts);

% Convert HTML to ReStructuredText
% Using Pandoc, a document converter
% http://johnmacfarlane.net/pandoc/

! pandoc -o ../README.almost.rst ../README.html
! sed -e 's/<#\(.\+\)>/<#id\1>/' ../README.almost.rst > ../README.rst
! rm ../README.almost.rst


% % To convert to Markdown, do this:
% if ~ exist('html2text.py', 'file')
%     ! wget http://www.aaronsw.com/2002/html2text/html2text.py
%     ! chmod a+rx html2text.py
% end
% 
% ! ./html2text.py ../README.html > ../README.markdown
% 

