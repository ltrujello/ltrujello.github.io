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

@pytest.fixture
def itemize_fixture():
    return r"""\begin{itemize}
\item[one] one
\item[two] two
\item[three] three
\end{itemize}"""

@pytest.fixture
def itemize_nested_fixture():
    return r"""\begin{itemize}
\item one 
\item two
\item \begin{itemize}
    \item one
    \item two
    \item \begin{itemize}
        \item foo bar
    \end{itemize}
\end{itemize}
\item three
\end{itemize}"""

@pytest.fixture
def description_fixture():
    return r"""\begin{description}
\item one 
\item two
\item three
\end{description}"""

@pytest.fixture
def itemize_description_fixture():
    return r"""\begin{itemize}
\item one 
\item two
\item \begin{description}
    \item one
    \item two
\end{description}
\item three
\end{itemize}"""

@pytest.fixture
def description_itemize_fixture():
    return r"""\begin{itemize}
\item[one] one 
\item[two] two
\item[three] \begin{description}
    \item[one] one
    \item[two] two
\end{description}
\item[four] four
\end{itemize}"""


def test_find_balanced_delimeters(tikzpicture_code):
    start_delimeter = "\\begin{tikzpicture}"
    end_delimeter= "\\end{tikzpicture}"
    assert find_balanced_delimeters(tikzpicture_code, start_delimeter, end_delimeter) == [(13, 641)]


def test_find_balanced_delimeters_2():
    start_delimeter = "\\textbf{"
    end_delimeter= "}"
    assert find_balanced_delimeters("\\textbf{presheaf of rings},", start_delimeter, end_delimeter) == [(0, 25)]

    
def test_find_balanced_delimeters_3():
    start_delimeter = "\\begin{itemize}"
    end_delimeter= "\\end{itemize}"
    content = r"""
\item one 
\item two
\item \begin{itemize}
    \item one
    \item two
\end{itemize}
\item three
""" 
    balanced_delimeters = find_balanced_delimeters(content, start_delimeter, end_delimeter) 
    start_ind, end_ind = balanced_delimeters[0]
    assert content[start_ind:start_ind+len(start_delimeter)] == start_delimeter 
    assert content[end_ind + 1 - len(end_delimeter):end_ind + 1] == end_delimeter
    

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
    
def test_extract_itemize_items_one_element():
    handler = Latex2Md(Path("foo"), Path("bar"), Path("foo"))
    handler.write_tikz_to_disk  = Mock()
    itemize_fixture = r"""\begin{itemize}
\item[one] one
\end{itemize}"""
    res = handler.extract_itemize_items(itemize_fixture)
    assert res == ["**one** one"]

def test_extract_itemize_items(itemize_fixture):
    handler = Latex2Md(Path("foo"), Path("bar"), Path("foo"))
    handler.write_tikz_to_disk  = Mock()

    res = handler.extract_itemize_items(itemize_fixture)
    assert res == ["**one** one", "**two** two", "**three** three"]

def test_extract_itemize_items_nested(itemize_nested_fixture):
    handler = Latex2Md(Path("foo"), Path("bar"), Path("foo"))
    handler.write_tikz_to_disk  = Mock()

    res = handler.extract_itemize_items(itemize_nested_fixture)
    print(res)
    assert res == ["one", "two", ["one", "two", ["foo bar"]], "three"]

def test_repl_itemize_env_one_element():
    handler = Latex2Md(Path("foo"), Path("bar"), Path("foo"))
    handler.write_tikz_to_disk  = Mock()
    itemize_fixture = r"""\begin{itemize}
\item[one] one
\end{itemize}"""
    res = handler.repl_itemize_env(itemize_fixture)
    assert res == """
* **one** one
"""

def test_repl_itemize_env(itemize_fixture):
    handler = Latex2Md(Path("foo"), Path("bar"), Path("foo"))
    handler.write_tikz_to_disk  = Mock()

    res = handler.repl_itemize_env(itemize_fixture)
    assert res == r"""
* **one** one
* **two** two
* **three** three
"""

def test_repl_itemize_env_nested(itemize_nested_fixture):
    handler = Latex2Md(Path("foo"), Path("bar"), Path("foo"))
    handler.write_tikz_to_disk  = Mock()
    itemize_fixture = r"""\begin{itemize}
\item[one] one
\end{itemize}"""
    res = handler.repl_itemize_env(itemize_nested_fixture)
    print(res)
    assert res == """
* one
* two
    * one
    * two
        * foo bar
* three
"""

def test_repl_itemize_env_nested(description_fixture):
    handler = Latex2Md(Path("foo"), Path("bar"), Path("foo"))
    handler.write_tikz_to_disk  = Mock()
    res = handler.repl_itemize_env(description_fixture)
    print(res)
    assert res == """
* one
* two
* three
"""

def test_repl_itemize_description_nested(itemize_description_fixture):
    handler = Latex2Md(Path("foo"), Path("bar"), Path("foo"))
    handler.write_tikz_to_disk  = Mock()
    res = handler.repl_itemize_env(itemize_description_fixture)
    print(res)
    assert res == """
* one
* two
    * one
    * two
* three
"""

def test_repl_description_itemize_nested(description_itemize_fixture):
    handler = Latex2Md(Path("foo"), Path("bar"), Path("foo"))
    handler.write_tikz_to_disk  = Mock()
    res = handler.repl_itemize_env(description_itemize_fixture)
    print(res)
    assert res == """
* **one** one
* **two** two
* **three**
    * **one** one
    * **two** two
* **four** four
"""


