'''
Uses pandoc (can change --mathjax to --katex for different math rendering).
In python_prototype, issue:
    pandoc -s --mathjax qmcpy/README.md -o ../html_from_readme/qmcpy.html
See also: http://www.flutterbys.com.au/stats/tut/tut17.3.html
'''

import os


def markdown_to_html(path, _dir):
    """
    Change README.md markdown files to HTML files. Also handle LaTeX \
    expressions in $...$.

    Args:
        path:
        _dir:

    Returns:

    """
    readme_path = path + "README.md"
    if os.path.isfile(readme_path):
        command = "pandoc -s -c https://www.w3schools.com/w3css/4/w3.css \
        --toc  --mathjax %s -o html_from_readme/%s.html" % (readme_path, _dir)
        os.system(command)
    dirs = os.listdir(path)
    for sub_dir in dirs:
        sub_path = path + sub_dir + "/"
        if os.path.isdir(sub_path):
            markdown_to_html(sub_path, sub_dir)


markdown_to_html("./", "QMCSoftware")
