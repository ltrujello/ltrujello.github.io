import pytest
from unittest.mock import Mock
from pathlib import Path
from latex_to_markdown import find_balanced_delimeters, Latex2Md

@pytest.fixture
def tikzpicture_code():
    return r"""
            \begin{tikzpicture}[black, scale = 1.5]
                %left triangle
                \draw (-4.1,0) -- (-3.1,1.5) node[above] {$1$};
                -0.5) {$rs$};

                        %middle center triangle.
                        \draw (-1,-3) -- (0,-1.5) node[above] {$2$};

                        \filldraw[red!50] (-1, -3) circle(0.5mm);

                \filldraw[red!50] (4.1, 0) circle (0.5mm);
                \begin{tikzpicture}
                meow
                \end{tikzpicture}
                %r^2 arrow
                \draw[thick, ->] (3.1,-0.1) to (3.1,-1) node at (3.4,

            \end{tikzpicture}
    """

def test_find_balanced_delimeters(tikzpicture_code):
    start_delimeter = "\\begin{tikzpicture}"
    end_delimeter= "\\end{tikzpicture}"
    assert find_balanced_delimeters(tikzpicture_code, start_delimeter, end_delimeter) == [(13, 641)]


def test_find_balanced_delimeters():
    start_delimeter = "\\textbf{"
    end_delimeter= "}"
    assert find_balanced_delimeters("\\textbf{presheaf of rings},", start_delimeter, end_delimeter) == [(0, 25)]

def test_repl_surround():
    start_delimeter = "\\textbf{"
    end_delimeter= "}"
    new_start_delimeter = "**"
    new_end_delimeter = "**"
    handler = Latex2Md(Path("foo"), Path("bar"), Path("foo"))
    assert handler.repl_surround("$} \omega: \\textbf{Trans}_P \to [0, \infty]$", start_delimeter, end_delimeter, new_start_delimeter, new_end_delimeter) == "$} \omega: **Trans**_P \to [0, \infty]$"

def test_repl_tikzpicture():
    handler = Latex2Md(Path("foo"), Path("bar"), Path("foo"))
    handler.write_tikz_to_disk  = Mock()
    tikz_code = r"""
        \begin{tikzpicture}[black, scale = 1.5]
            %left triangle
            \draw (-4.1,0) -- (-3.1,1.5) node[above] {$1$};
            -0.5) {$rs$};

                    %middle center triangle.
                    \draw (-1,-3) -- (0,-1.5) node[above] {$2$};

                    \filldraw[red!50] (-1, -3) circle(0.5mm);

            \filldraw[red!50] (4.1, 0) circle (0.5mm);
            \begin{tikzpicture}
            meow
            \end{tikzpicture}
            %r^2 arrow
            \draw[thick, ->] (3.1,-0.1) to (3.1,-1) node at (3.4,

        \end{tikzpicture}
        thing in middle
            \begin{tikzpicture}
            meow
            \end{tikzpicture}
        thing in middle again
            \begin{tikzpicture}
            meow
            \end{tikzpicture}
    """
    res = handler.repl_tikzpicture(tikz_code, 1, 1)
    assert res == '\n        \n<img src="../../../png/bar/chapter_1/tikz_code_1_0.png" width="99%" style="display: block; margin-left: auto; margin-right: auto;"/>\n\n        thing in middle\n            \n<img src="../../../png/bar/chapter_1/tikz_code_1_1.png" width="99%" style="display: block; margin-left: auto; margin-right: auto;"/>\n\n        thing in middle again\n            \n<img src="../../../png/bar/chapter_1/tikz_code_1_2.png" width="99%" style="display: block; margin-left: auto; margin-right: auto;"/>\n\n    ' 
    assert handler.num_figures == 3
    



