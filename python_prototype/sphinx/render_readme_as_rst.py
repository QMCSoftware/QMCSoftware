"""
Use pandoc to recursively change markdown files to restructured text files.

Note: can change --mathjax to --katex for different math rendering.

In python_prototype, issue:
    pandoc -s --mathjax qmcpy/README.md -o ../markdown_to_rst/qmcpy.html

See also: http://www.flutterbys.com.au/stats/tut/tut17.3.html

"""

import os

def markdown_to_rst(path, _dir):
    """
    Change README.md markdown files to .rst files. Also handle LaTeX \
    expressions in $...$.

    Args:
        path (str): Top-level directory that contains README.md and
            subdirectories that may also contain README.md
        _dir (str): output filename of rst for the top-level README.md

    Returns: None

    """
    readme_path = path + "README.md"
    rst_ouput_dir = "python_prototype/sphinx/markdown_to_rst"
    if os.path.isfile(readme_path):
        command = "pandoc -s -c https://www.w3schools.com/w3css/4/w3.css \
        --toc --toc-depth=1 --mathjax %s -o %s/%s.rst" % (readme_path, rst_ouput_dir, _dir)
        os.system(command)
    dirs = os.listdir(path)
    for sub_dir in dirs:
        sub_path = path + sub_dir + "/"
        if os.path.isdir(sub_path):
            markdown_to_rst(sub_path, sub_dir)


if __name__ == '__main__':
    markdown_to_rst("./", "QMCSoftware")
