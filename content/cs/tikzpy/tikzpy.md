<!-- title: Tikz-Python -->
<!-- syntax_highlighting: on -->
<!-- preview_img: /cs/tikzpy/imgs/tikzpy_preview.png -->
<!-- date: 2021-09-10 -->

An object-oriented Python approach towards providing a giant wrapper for Tikz code, with the goal of streamlining the process of creating complex figures for TeX documents.

## Requirements
This requires Python 3.7+. Additionally, you need an up-to-date version of a LaTeX that has both the `tikz` library and `latexmk`. 

If you already have LaTeX or a related TeX engine, you most definitely have `tikz`. You also probably have `latexmk`. If you're not sure, run `latexmk --version` from the command line and observe the output.

To run the tests, you need the python packages `numpy`, [`pytest`](https://github.com/pytest-dev/pytest) and [`pytest-order`](https://github.com/pytest-dev/pytest-orderhttps://github.com/pytest-dev/pytest-order). These will be installed for you if they are missing.

## Installation 
You can install Tikzpy as follows.
```bash 
$ git clone https://github.com/ltrujello/Tikz-Python
$ cd Tikz-Python
$ pip install --use-feature=in-tree-build .
```
Check that everything is working normally by running
```bash
$ cd tests
$ pytest
```
All test cases should pass. Let me know if that is not the case.

## Troubleshooting
If pip tries to tell you `no such option: --use-feature`, then that means you need to upgrade your pip which you can do via `pip install --upgrade pip`. If you don't want to upgrade your pip, then simply remove the `--use-feature` flag, at which point pip may scream a warning to you about in tree builds. 

If pip gives you a truly nonsense error with some keywords like `exit status 1` or `check the logs`, running `pip install -U setuptools` should do the trick.

## How to Use: Basics
An example of this package in action is below. 
```python
from tikzpy import TikzPicture  # Import the class TikzPicture

tikz = TikzPicture()
tikz.circle((0, 0), 3, options="thin, fill=orange!15")

arc_one = tikz.arc((3, 0), 0, 180, x_radius=3, y_radius=1.5, options="dashed")
arc_two = tikz.arc((-3, 0), 180, 360, x_radius=3, y_radius=1.5)

tikz.write()  # Writes the Tikz code into a file
tikz.show()  # Displays a pdf of the drawing to the user
```
which produces
<img src="/cs/tikzpy/imgs/basic.png"/> 

We explain line-by-line the above code.

* `from tikzpy import TikzPicture` imports the `TikzPicture` class from the `tikzpy` package. 

* The second line of code is analogous to the TeX code `\begin{tikzpicture}` and `\end{tikzpicture}`. The variable `tikz` is now a tikz environment, specifically an instance of the class `TikzPicture`, and we can now append drawings to it.

* The third, fourth, and fifth lines draw a filled circle and two elliptic arcs, which give the illusion of a sphere.

* In the last two lines, `write()` writes all of our tikz code into a file located at `tikz_code/tikz_code.tex`. The call `show()` immediately displays the PDF of the drawing to the user.

## Examples
We introduce more examples of `tikzpy`, starting from very basic to more complicated usages.

### Example: Line and two nodes
Suppose I want to create a line and two labels at the ends. The code below achieves this
```python
from tikzpy import TikzPicture

tikz = TikzPicture()
line = tikz.line((0, 0), (1, 1), options="thick, blue, o-o")
start_node = tikz.node(line.start, options="below", text="Start!")
end_node = tikz.node(line.end, options="above", text="End!")
```
and produces

<img src="/cs/tikzpy/imgs/line_and_two_nodes.png"/> 

Saving the line as a variable `line` allows us to pass in `line.start` and `line.end` into the node positions, so we don't have to type out the exact coordinates. 
This is because lines, nodes, etc. are class instances with useful attributes: 
```python
>>> line.start
(0,0)
>>> line.end
(1,1)
>>> start_node.text
"Start!"
```

### Example: Circles
In this example, we use a for loop to draw a pattern of circles. 

This example 
demonstates how Pythons `for` loop is a lot less messier than the `\foreach` loop provided in Tikz via TeX. (It is also more powerful; for example, Tikz with TeX alone guesses your step size, and hence it cannot effectively [loop over two different sequences at the same time](https://tex.stackexchange.com/questions/171426/increments-in-foreach-loop-with-two-variables-tikz)).

```python
import numpy as np
from tikzpy import TikzPicture

tikz = TikzPicture(center=True)

for i in np.linspace(0, 1, 30): # Grab 30 equidistant points in [0, 1]
    point = (np.sin(2 * np.pi * i), np.cos(2 * np.pi * i))

    # Create four circles of different radii with center located at point
    tikz.circle(point, 2, "ProcessBlue")
    tikz.circle(point, 2.2, "ForestGreen")
    tikz.circle(point, 2.4, "red")  # xcolor Red is very ugly
    tikz.circle(point, 2.6, "Purple")
```
The above code then produces

<img src="/cs/tikzpy/imgs/circles.png"/>


### Example: Roots of Unity 
In this example, we draw the 13 [roots of unity](https://en.wikipedia.org/wiki/Root_of_unity). 

If we wanted to normally do this in TeX, we'd
probably have to spend 30 minutes reading some manual about how TeX handles basic math. With Python, we can just use the `math` library and make intuitive computations to quickly build a function that displays the nth roots of unity.
```python
from math import pi, sin, cos
from tikzpy import TikzPicture

tikz = TikzPicture()
n = 13 # Let's see the 13 roots of unity
scale = 5

for i in range(n):
    theta = (2 * pi * i) / n
    x, y = scale * cos(theta), scale * sin(theta)
    content = f"$e^{{ (2 \cdot \pi \cdot {i})/ {n} }}$"

    # Draw line to nth root of unity
    tikz.line((0, 0), (x, y), options="-o")

    if 0 <= theta <= pi:
        node_option = "above"
    else:
        node_option = "below"

    # Label the nth root of unity
    tikz.node((x, y), options=node_option, text=content)

```
Which generates: 

<img src="/cs/tikzpy/imgs/roots_of_unity.png"/>

We will see in the examples that follow how imported Python libraries can alllow us to quickly (and efficiently, this is really important) make more sophisticated Tikz pictures. 

### Example: General Ven Diagrams 
In the [source here](https://github.com/ltrujello/Tikz-Python/blob/main/examples/ven_diagrams/intersections_scope_clip.py), we use the python library `itertools.combinations` to create a function which takes in an arbitrary number of 2D Tikz figures and colors each and every single intersection. 

For example, suppose we arrange nine circles in a 3 x 3 grid. Plugging these nine circles in, we generate the image below.

<img src="/cs/tikzpy/imgs/intersection_circles.png"/>

As another example, we can create three different overlapping topological blobs and then plug them into the function to obtain

<img src="/cs/tikzpy/imgs/intersection_blobs.png"/>

(Both examples are initialized in [the source](https://github.com/ltrujello/Tikz-Python/blob/main/examples/ven_diagrams/intersections_scope_clip.py) for testing.)
As one might guess, this function is useful for creating topological figures, as manually writing all of the `\scope` and `\clip` commands to create such images is pretty tedious.

### Example: Barycentric subdivision
In [the source here](https://github.com/ltrujello/Tikz-Python/blob/main/examples/barycentric/barycentric.py), we create a function that allows us to generate the the n-th barycentric subdivision of a triangle. 

<img src="/cs/tikzpy/imgs/barycentric.png"/>

### Example: Cantor function
In [the source here](https://github.com/ltrujello/Tikz-Python/blob/main/examples/cantor/cantor.py), we plot the Cantor function by performing recursion. It is clear from this [TeX Stackexchange question](https://tex.stackexchange.com/questions/241622/plotting-the-cantor-function) that TeX alone cannot do this, as most answers rely on external programs to generate the data. 

<img src="/cs/tikzpy/imgs/cantor.png"/>

### Example: Symbolic Intergation
In [the source here](https://github.com/ltrujello/Tikz-Python/blob/main/examples/symbolic_integration/integrate_and_plot.py), we use `numpy` and `sympy` to very simply perform symbolic integration. The result is a function which plots and labels the n-order integrals of any function. For example, the output of `x**2` (the polynomial x^2) generates the image below. 

<img src="/cs/tikzpy/imgs/integration_ex.png"/>

### Example: Cone over a Projective Variety
In [the source here](https://github.com/ltrujello/Tikz-Python/blob/main/examples/projective_cone/projective_cone.py), we use `numpy` to create an image which illustrates the concept of an affine cone over a projective variety. In the case of a curve Y in P^2, the cone C(Y) is a surface in A^3. 

The image that this drawing was modeled after appears in Exercise 2.10 of Hartshorne's Algebraic Geometry.

<img src="/cs/tikzpy/imgs/projective_cone.png"/>

### Example: Lorenz System
In [the source here](https://github.com/ltrujello/Tikz-Python/blob/main/examples/lorenz/lorenz.py), we use `numpy` and `scipy` to solve ODEs and plot the Lorenz system. This is made possible since `tikz_py` also supports 3D. 

<img src="/cs/tikzpy/imgs/lorenz_ex.png"/>

### Example: Tikz Styles
`tikzpy` supports the creation of any `\tikzset`, a feature of Tikz that saves users a great deal of time. You can save your tikz styles in a .py file instead of copying and pasting all the time. 

Even if you don't want to make such settings, there are useful `\tikzset` styles that are preloaded in `tikzpy`. One particular is the very popular tikzset authored by Paul Gaborit [in this TeX stackexchange question](https://tex.stackexchange.com/questions/3161/tikz-how-to-draw-an-arrow-in-the-middle-of-the-line). Using such settings, we create these pictures, which illustrate Cauchy's Residue Theorem.
[The source here](https://github.com/ltrujello/Tikz-Python/blob/main/examples/cauchy_residue_thm/cauchy_residue_thm.py) produces 

<img src="/cs/tikzpy/imgs/cauchy_residue_thm_ex.png"/>

while [the source here](https://github.com/ltrujello/Tikz-Python/blob/main/examples/cauchy_residue_thm/cauchy_residue_thm_arc.py) produces 

<img src="/cs/tikzpy/imgs/cauchy_residue_thm_arc_ex.png"/>

### Example: Linear Transformations
Recall a 3x2 matrix is a linear transformation from R^2 to R^3. Using such an interpretation, we create a function in [the source here](https://github.com/ltrujello/Tikz-Python/blob/main/examples/linear_transformations/linear_transformations.py) which plots the image of a 3x2 matrix. The input is in the form of a `numpy.array`. 

For example, plugging the array `np.array([[0, 1], [1, 1], [0, 1]])` into the source produces 

<img src="/cs/tikzpy/imgs/linear_transformation_ex_1.png"/>

while plugging the array `np.array([[2, 0], [1, 1], [1, 1]])` into the source produces 

<img src="/cs/tikzpy/imgs/linear_transformation_ex_2.png"/>

### Example: Projecting R^1 onto S^1
In [the source here](https://github.com/ltrujello/Tikz-Python/blob/main/examples/spiral/spiral.py), we use `numpy` to illustrate the projection of R^1 onto S^1. Creating this figure in Tex alone is nontrivial, as one must create white space at self intersections to illustrate crossovers. Existing tikz solutions cannot take care of this, but the flexible logical operators of Python allow one to achieve it. 

<img src="/cs/tikzpy/imgs/spiral.png"/>

### Example: Polar Coordinates
In [the source here](https://github.com/ltrujello/Tikz-Python/blob/main/examples/polar/polar.py), we illustrate the concept of polar coordiantes by demonstrating how a sine curve is mapped into polar coordinates. This example should be compared to the more complex answers in this [TeX Stackexchange question](https://tex.stackexchange.com/questions/594231/make-rainbow-coloured-bullets-to-show-points) which seeks a similar result. 

<img src="/cs/tikzpy/imgs/polar.png"/>

### Example: Blowup at a point
In [the source here](https://github.com/ltrujello/Tikz-Python/blob/main/examples/blowup/blowup.py), we illustrate the blowup of a point, a construction in algebraic geometry. This picture was created in 5 minutes and in half the lines of code compared to [this popular TeX stackexchange answer](https://tex.stackexchange.com/a/158762/195136), which uses quite convoluted, C-like Asymptote code.

<img src="/cs/tikzpy/imgs/blowup_ex.png"/>

# Class: `TikzPicture`
Initialize an object of this class as below
```python
from tikzpy import TikzPicture

tikz = TikzPicture(center, options)
```


| Parameter       | Description                                                              | Default |
| --------------- | ------------------------------------------------------------------------ | ------- |
| `center` (bool) | True if you would like your tikzpicture to be centered, false otherwise. | `False` |
| `options` (str) | A string containing valid Tikz options.                                  | `""`    |


## Methods  
### `TikzPicture.write()`
Writes the currently recorded Tikz code into the .tex file located in the relative directory `tikz_code/tikz_code.tex`. If it does not exist, it is automatically created. 

**Question:** If I call `tikz.write()` twice on accident, won't that accidentally add duplicate code? No! That would be annoying. You can continue editing even after you call `write` so that you may periodically view your Tikz picture while you build it.

For example, suppose I want to draw a blue circle.
```python
>>> tikz = TikzPicture()
>>> circle_1 = tikz.circle((0,0), 2, options = "fill=Blue, opacity=0.5") # Draws a blue circle 
>>> tikz.write() # Write it 
```
<img src="/cs/tikzpy/imgs/tikz_write_ex_1.png"/>

I called `.write()` which writes the code for a blue circle. In the same instance, I can add `circle_2`, a red circle, to the picture. I can also update the center of `circle_1`.
```python
>>> circle_2 = tikz.circle((1,1), 2, options = "fill=red, opacity=0.5") # I want another circle...
>>> circle_1.center = (2,2) # I want to change my other circle's center...
>>> tikz.write() # Write it 
>>> tikz # The resulting tikzcode. We get what we'd expect  
... \begin{tikzpicture}[]% TikzPython id = (1) 
	\draw[fill=Blue] (2, 2) circle (2cm);
	\draw[fill=Red] (1, 1) circle (2cm);
\end{tikzpicture}
```
<img src="/cs/tikzpy/imgs/tikz_write_ex_2.png"/>

This feature, in combination with `.remove()` and `.show()` (see below), allows you to gradually build and view a TikzPicture quite painlessly.

### `TikzPicture.remove(draw_obj)`
Removes a drawing object, such as a line, from a TikzPicture. Here, we draw an arc and a line. Then, we remove the line.

```python
>>> tikz = TikzPicture()
>>> line = tikz.line((0,0), (1,1), options = "Blue") # Draws a line 
>>> arc = tikz.arc((0,0), 45, 90, 3) # Draws an arc
>>> tikz 
... \begin{tikzpicture}[]% TikzPython id = (1)
    \draw[Blue] (0, 0) -- (1,1); # The line
    \draw (0, 0) arc (45:90:3cm);
\end{tikzpicture}
>>> tikz.remove(line) # The line is removed
>>> tikz 
... \begin{tikzpicture}[]% TikzPython id = (1) 
	\draw (0, 0) arc (45:90:3cm);
\end{tikzpicture}
```

### `TikzPicture.draw(draw_obj)`
Draws a drawing object, such as a line, onto the TikzPicture. Sometimes, we want to construct a drawing object before we want to draw it onto the tikzpicture. 
```python
>>> from tikzpy import TikzPicture, Line, Circle
>>> line = Line((0,0) (1,0), to_options = "to[bend right = 30]")
>>> end_c = Circle(line.start, radius = 0.2)
>>> start_c = Circle(line.end, radius = 0.2)
>>> tikz = TikzPicture()
>>> tikz.draw(line, end_c, start_c) # The line and circles are now drawn
```
Here we use `Line` and `Circle` classes to construct a line and circle. These objects have no relation to our image until we draw them with `tikz.draw`.


### `TikzPicture.show()`
Compiles the tikz code and pulls up a PDF of the current drawing to the user in your browser (may default to your PDF viewer). Of course, execute `TikzPicture.write()` prior in order to view your latest changes. 

### `TikzPicture.add_command(str)`
Manually add a valid string of Tikz code to the environment. This is for the off-chance that the user would rather manually type something into their tikzpicture.

### `TikzPicture.drawing_objects()`
Returns a list of the currently appended drawing objects in the TikzPicture object. The list is ordered chronologically from oldest to most recently added.


# Colors
Coloring Tikz pictures in TeX tends to be annoying. A goal of this has been to make it as easy as possible to color Tikz pictures.

- One is free to use whatever colors they like, but `\usepackage[dvipnames]{xcolor}` is loaded in the TeX document which compiles the Tikz code. 

- There is also a global function `tikzpy.rgb(r, g, b)` which can be called to color a Tikz object by RGB values. For example, 
```python
from tikzpy import TikzPicture
from tikzpy import rgb

>>> tikz = TikzPicture()
>>> line =  tikz.line((1,2), (4,3), options = "color=" + rgb(253, 0, 0))
>>> rectangle = tikz.rectangle( (0,0), (5,5)), options = "fill=" + rgb(120, 0, 120))
```

- A wrapper function `rainbow_colors` uses the `rgb` to provide rainbow colors. The function takes in any integer, and grabs a rainbow color, computing a modulo operation if necessary  (hence, any integer is valid). 
```python
from tikzpy import TikzPicture
from tikzpy import rainbow_colors

>>> tikz = TikzPicture()
>>> for i in range(0, 20):
        circle = tikz.circle((i/20, 3 - i**2/20), 3)
        circle.options = "opacity = 0.7, fill = " + rainbow_colors(i)
```


# Class: `Line`
There are two ways to initalize a line object. We've already seen this way:
```python
from tikzpy import TikzPicture

tikz = TikzPicture()
line = tikz.line(start, end, options, to_options, control_pts, action) # A line is created and drawn
```
This is a "quick draw": we simultaneously create a line instance *and* draw it. 
But we can also initailize a line in its own right:
```python
from tikzpy import Line

line = Line(start, end, options, to_options, control_pts, action) # We create a line
```
We can add this line in later to see it whenever we like via `tikz.draw(line)`. 

Note: A natural question is: Why the two ways? This is because sometimes we want to obtain information from a drawing object to perform some calculations *before* we decide to actually draw it. One familiar with Tikz will realize that this is analagous to the `\path` command in Tikz, which is often very useful. 

| Parameter            | Description                                                                                               | Default   |
| -------------------- | --------------------------------------------------------------------------------------------------------- | --------- |
| `start` (tuple)      | Pair of floats representing the start of the line                                                         |
| `end` (tuple)        | Pair of floats representing the end of the line                                                           |
| `options` (str)      | String containing valid Tikz drawing options, e.g. "Blue"                                                 | `""`      |
| `to_options` (str)   | String containing Tikz specifications for connecting the start to the end (e.g. `"to [bend right = 45]"`) | "--"      |
| `control_pts` (list) | List of control points for the line                                                                       | `[]`      |
| `action` (str)       | An action to perform with plot (e.g., `\draw`, `\fill`, `\filldraw`, `\path`)                             | `"\draw"` |

## Examples
We've already seen an example of this class. Here's another.
```python
from tikzpy import TikzPicture

tikz = TikzPicture()
tikz.line((0, 0), (4, 0), options="->", control_pts=[(1, 1), (3, -1)])
```
produces the line 

<img src="/cs/tikzpy/imgs/line_ex_1.png"/> 

## Methods 
### `Line.shift(xshift, yshift)`
Shifts the current line by amount `xshift` in the x-direction and `yshift` in the y-direction. For example, we can shift in a for loop as below
```python
from tikzpy import TikzPicture, Line

tikz = TikzPicture(center=True)

line_template = Line((0, 0), (4, 0), control_pts=[(1, 1), (3, -1)])
for i in range(0, 10):
    line = line_template.copy() # Make a copy 
    line.shift(0, (5 - i) / 4) # Shift the copy
    line.options = f"color={rainbow_colors(i)}, o->" #Specify options
    tikz.draw(line)
```
which produces the set of lines 

<img src="/cs/tikzpy/imgs/line_ex_2.png"/>

### `Line.scale(scale)`
Scales a line by an amount `scale`, usually a python float. 

### `Line.rotate(angle, about_pt = None, radians = True)`
Rotates a line counterclockwise by angle `angle` relative to the point `about_pt`. One can specify their angle units via the boolean `radians`. If `about_pt` is not specified, the default is to rotate the line about its midpoint.

Here's an example of both `.scale` and `rotate` being used.



# Class: `PlotCoordinates`
Initialize an object of the class as below:
```python
from tikzpy import TikzPicture

tikz = TikzPicture()
plot = tikz.plot_coordinates(points, options, plot_options, action)
```
which simultaneously creates and draws a `PlotCoordinates` object. Or more simply, we can create an instance as:
```python
from tikzpy import PlotCoordinates

plot = PlotCoordinates(points, options, plot_options, action)
```
which we can add to our picture later via `tikz.draw(plot)`.

| Parameter            | Description                                                                            | Default   |
| -------------------- | -------------------------------------------------------------------------------------- | --------- |
| `points` (list)      | A list of tuples (x, y) representing coordinates that one wishes to create a plot for. |
| `options` (str)      | A string of valid Tikz drawing options.                                                | `""`      |
| `plot_options` (str) | A string of valid Tikz plotting options                                                | `""`      |
| `action` (str)       | An action to perform with the line (e.g., `\draw`, `\fill`, `\filldraw`, `\path`)      | `"\draw"` |

This class is analagous to the Tikz command `\draw plot coordinates{...};`.

## Examples
Introducing examples of `PlotCoordinates` gives us an opportunity to illustrate the optional parameter `action`. By default, `action` is `"draw"` (analogous to `\draw` in Tikz) so the code below
```python
from tikzpy import TikzPicture

tikz = TikzPicture()
points = [(2, 2), (4, 0), (1, -3), (-2, -1), (-1, 3)]
plot = tikz.plot_coordinates(points) # action="draw" by default
plot.plot_options = "smooth cycle, tension = 0.5"
```
produces the image 

<img src="/cs/tikzpy/imgs/plotcoordinates_ex_1.png"/>

Alternatively we can set `action = "fill"` (analogous to `\fill` in Tikz) as in the code below
```python
from tikzpy import TikzPicture

tikz = TikzPicture()
points = [(2, 2), (4, 0), (1, -3), (-2, -1), (-1, 3)]
plot = tikz.plot_coordinates(points, options="Blue", action="fill")
plot.plot_options = "smooth cycle, tension = 0.5"
```
to produce the image

<img src="/cs/tikzpy/imgs/plotcoordinates_ex_2.png"/>

If we want both, we can set `action = "filldraw"` (analogous to `\filldraw` in Tikz)
```python
from tikzpy import TikzPicture

tikz = TikzPicture()
points = [(2, 2), (4, 0), (1, -3), (-2, -1), (-1, 3)]
plot = tikz.plot_coordinates(points, options="Blue", action="filldraw")
plot.options = "fill=ProcessBlue!50"
plot.plot_options = "smooth cycle, tension = 0.5"
```
which produces. 
<img src="/cs/tikzpy/imgs/plotcoordinates_ex_3.png"/>

Finally, we can set `action = "path"` (analogous to `\path` in Tikz), but as one would expect this doesn't draw anything. 

## Methods

`PlotCoordinates` has methods `.shift()`, `.scale`, and `.rotate`, similar to the class `Line`, and the parameters behave similarly. These methods are more interestingly used on `PlotCoordinates` than on `Line`. For example, the code
```python
from tikzpy import TikzPicture

tikz = TikzPicture()
points = [(14.4, 3.2), (16.0, 3.6), (16.8, 4.8), (16.0, 6.8), (16.4, 8.8), (13.6, 8.8), (12.4, 7.6), (12.8, 5.6), (12.4, 3.6)]

for i in range(0, 20):
    options = f"fill = {rainbow_colors(i)}, opacity = 0.7"
    # Requires \usetikzlibrary{hobby} here
    plot_options = "smooth, tension=.5, closed hobby"
    plot = tikz.plot_coordinates(points, options, plot_options)
    plot.scale((20 - i) / 20) # Shrink it 
    plot.rotate(15 * i) # Rotate it
```
generates the image

<img src="/cs/tikzpy/imgs/PlotCoords_rotate_Example.png"/>


# Class: `Circle`
Initialize an object of the class as below:
```python
from tikzpy import TikzPicture

tikz = TikzPicture()
circle = tikz.circle(center, radius, options, action)
```
which creates a `Circle` object and draws it. Alternatively, we can initalize more simply as 
```python
from tikzpy import Circle

circle = Circle(center, radius, options, action)
```
and later draw this via `tikz.draw(circle)`.

| Parameter        | Description                                                                         | Default   |
| ---------------- | ----------------------------------------------------------------------------------- | --------- |
| `center` (tuple) | A tuple (x, y) of floats representing the coordinates of the center of the circle.  |
| `radius` (float) | Length (in cm) of the radius. (By the way, all lengths are taken in cm).            |
| `options` (str)  | String containing valid Tikz drawing options (e.g, "Blue")                          | `""`      |
| `action` (str)   | An action to perform with the circle (e.g., `\draw`, `\fill`, `\filldraw`, `\path`) | `"\draw"` |


## Examples
Here we create several circles, making use of the `action` parameter. 
```python
from tikzpy import TikzPicture

tikz = TikzPicture()
tikz.circle((0, 0), 1.25) #action="draw" by default
tikz.line((0, 0), (0, 1.25), options="dashed")
tikz.circle((3, 0), 1, options="thick, fill=red!60", action="filldraw")
tikz.circle((6, 0), 1.25, options="Green!50", action="fill")
```

<img src="/cs/tikzpy/imgs/circle_ex_1.png"/>

We can also use circles to create the [Hawaiian Earing](https://en.wikipedia.org/wiki/Hawaiian_earring).

```python
from tikzpy import TikzPicture

tikz = TikzPicture()

radius = 5
for i in range(1, 60):
    n = radius / i
    tikz.circle((n, 0), n)
```
<img src="/cs/tikzpy/imgs/circle_ex_2.png"/>


## Methods
`Circle` has access to methods `.shift()`, `.scale()`, `.rotate()`, which behave as one would expect and takes in parameters as described before.


# Class: `Node`
Initialize an object of the class as below:
```python
from tikzpy import TikzPicture

tikz = TikzPicture()
node = tikz.node(position, options, text)
```
which creates a `Node` object and draws it. We can also intiailize a 
node object directly with
```python
from tikzpy import Node

node = Node(position, options, text)
```
We can then add the node later via `tikz.draw(node)`.

| Parameter          | Description                                                                            | Default |
| ------------------ | -------------------------------------------------------------------------------------- | ------- |
| `position` (tuple) | A tuple (x, y) of floats representing the position of the node                         |
| `options` (str)    | String containing valid Tikz node options (e.g., "Above")                              | `""`    |
| `text` (str)       | A string containing content, such as text or LaTeX code, to be displayed with the node | `""`    |

## Examples
Here we use some nodes to label a figure explaining the logarithm branch cut
```python
from tikzpy import TikzPicture

tikz = TikzPicture()
# x,y axes
tikz.line((-4, 0), (4, 0), options="Gray!40, ->")
tikz.line((0, -4), (0, 4), options="Gray!40, ->")
# Cut
tikz.line((-4, 0), (0, 0), options="thick")
# Line out
tikz.line((0, 0), (1.414, 1.414), options="-o")
tikz.arc((1, 0), 0, 45, radius=1, options="dashed")

# Labels
tikz.node((3.6, -0.2), text="$x$")
tikz.node((-0.24, 3.53), text="$iy$")
tikz.node((1.3, 0.4), text="$\\theta$")
tikz.node((2.1, 1.7), text="$z = re^{i\\theta}$")
tikz.node((-2, 0.3), text="Cut")
```
which produces
<img src="/cs/tikzpy/imgs/node_ex_1.png"/>

Here's another example of usings nodes to illustrate the concept of a multivariable function.
```python
from tikzpy import TikzPicture

row_1 = TikzPicture()

# Lines and rectangles
row_1.line((0, 0), (2, 0), options="->")
row_1.rectangle((2, -0.5), (4, 0.5))
row_1.line((4, 0), (6, 0), options="->")
# Labels
row_1.node((-1.2, 0), text="$(x_1, \dots, x_n)$")
row_1.node((1, 0.3), text="input")
row_1.node((3, 0), text="$f$")
row_1.node((5, 0.3), text="output")
row_1.node((7.3, 0), text="$f(x_1, \dots, x_n)$")
```

<img src="/cs/tikzpy/imgs/node_ex_2.png"/>

## Methods

`Node` has access to methods `.shift()`, `.scale()`, `.rotate()`, which behave as one would expect and takes in parameters as described before.

# Class: `Rectangle`
Initialize an object of the class as below:
```python
from tikzpy import TikzPicture

tikz = TikzPicture()
rectangle = tikz.rectangle(left_corner, right_corner, options, action)
```
which creates a `Rectangle` object and draws it. We can also write
```python
from tikzpy import Rectangle

rectangle = Rectangle(left_corner, right_corner, options, action)
```
to create an instance, and later draw via `tikz.draw(Rectangle)`.

| Parameter              | Description                                                                            | Default   |
| ---------------------- | -------------------------------------------------------------------------------------- | --------- |
| `left_corner`  (tuple) | A tuple (x, y) of floats representing the position of the node.                        |
| `right_corner` (str)   | String containing valid Tikz node options (e.g., "above")                              | `""`      |
| `options` (str)        | A string containing valid Tikz draw optins, (e.g, "fill = Blue").                      | `""`      |
| `action` (str)         | An action to perform with the rectangle (e.g., `\draw`, `\fill`, `\filldraw`, `\path`) | `"\draw"` |

## Example
Rectangles are often used as a background to many figures; in this case, 
we create a fancy colored background.

```python
from tikzpy import TikzPicture

tikz = TikzPicture()

tikz.rectangle((-3.5, -2.5), (4.5, 2.5), options="rounded corners, Yellow!30",action="filldraw")
# Params
r = 2
n_nodes = 7
nodes = []
# Draw the nodes
for i in range(1, n_nodes + 1):
    angle = 2 * math.pi * i / n_nodes 
    x = r * math.cos(angle)
    y = r * math.sin(angle)
    node = tikz.node((x, y), text=f"$A_{{{i}}}$")
    nodes.append(node)

# Draw the lines between the nodes
for i in range(len(nodes)):
    start = nodes[i].position
    end = nodes[(i + 1) % len(nodes)].position
    tikz.line(start, end, options="->, shorten >= 10pt, shorten <=10pt")
```

<img src="/cs/tikzpy/imgs/rectangle_ex_1.png"/>


## Methods
`Rectangle` has access to methods `.shift()`, `.scale()`, `.rotate()`, which behave as one would expect and takes in parameters as described before.

# Class: `Ellipse`
Initialize an object of the class as below:
```python
from tikzpy import TikzPicture

tikz = TikzPicture()
ellipse = tikz.ellipse(center, x_axis, y_axis, options, action)
```
which creates an `Ellipse` object and draws it. We can also write
```python
from tikzpy import Ellipse

ellipse = Ellipse(center, x_axis, y_axis, options, action)
```
and draw this later to the Tikz picture via `tikz.draw(ellipse)`.

| Parameter        | Description                                                                          | Default   |
| ---------------- | ------------------------------------------------------------------------------------ | --------- |
| `center` (tuple) | Pair of floats representing the center of the ellipse                                |
| `x_axis` (float) | The length (in cm) of the horizontal axis of the ellipse                             |
| `y_axis` (float) | The length (in cm) of the vertical axis of the ellipse                               |
| `action` (str)   | An action to perform with the ellipse (e.g., `\draw`, `\fill`, `\filldraw`, `\path`) | `"\draw"` |


## Example
Here we draw and ellipse and define the major and minors axes.
```python
from tikzpy import TikzPicture

tikz = TikzPicture()

# x,y axes
tikz.line((-5, 0), (5, 0), options="Gray!40, ->")
tikz.line((0, -5), (0, 5), options="Gray!40, ->")
# Ellipse
ellipse = tikz.ellipse(
    (0, 0), 4, 3, options="fill=ProcessBlue!70, opacity=0.4", action="filldraw"
)
# Labels
h_line = tikz.line((0, 0), (ellipse.x_axis, 0), options="thick, dashed, ->")
v_line = tikz.line((0, 0), (0, ellipse.y_axis), options="thick, dashed, ->")
tikz.node(h_line.midpoint, options="below", text="Major")
tikz.node(v_line.midpoint, options="left", text="Minor")
```

<img src="/cs/tikzpy/imgs/ellipse_ex_1.png" height = 250/>


## Methods
`Ellipse` has access to methods `.shift()`, `.scale()`, `.rotate()`, which behave as one would expect and takes in parameters as described before.


# Class: `Arc`
Initialize an object of the class as below:

```python
from tikzpy import TikzPicture

tikz = TikzPicture()
arc = tikz.arc(center, start_angle, end_angle, radius, options, radians, action)
```
which creates an `Arc` object and draws it. Again, we can 
also initalize an instance with
```python
from tikzpy import Arc

arc = Arc(center, start_angle, end_angle, radius, options, radians, action)
```
which we can draw later via `tikz.draw(arc)`.

| Parameter                | Description                                                                                                                                      | Default   |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ | --------- |
| `position` (tuple)       | Pair of points representing either the center of the arc or the point at which it should begin drawing (see `draw_from_start`                    |
| `start_angle` (float)    | The angle (relative to the horizontal) of the start of the arc                                                                                   |
| `end_angle` (float)      | The angle (relative to the horizontal) of the end of the arc                                                                                     |
| `radius` (float)         | The radius (in cm) of the arc. If specified, `x_radius` and `y_radius` cannot be specified.                                                      |
| `x_radius` (float)       | The x radius (in cm) of the arc. In this case, `y_radius` must also be specified.                                                                |
| `y_radius` (float)       | The y radius (in cm) of the arc. In this case, the `x_radius` must also be specified.                                                            |
| `options` (str)          | A string of containing valid Tikz arc options                                                                                                    | `""`      |
| `radians` (bool)         | `True` if angles are in radians, `False` otherwise                                                                                               | `False`   |
| `draw_from_start` (bool) | `True` if `position` represents the point at which the arc should begin drawing. `False` if `position` represents the center of the desired arc. | `True`    |
| `action` (str)           | An action to perform with the arc (e.g., `\draw`, `\fill`, `\filldraw`, `\path`)                                                                 | `"\draw"` |

## A few comments...
This class not only provides a wrapper to draw arcs, but it also fixes a few things that Tikz's `\draw arc` command simply gets wrong and confuses users with.

1. With Tikz in TeX, to draw a circular arc one must specify `start_angle` and `end_angle`. These make sense: they are the start and end angles of the arc relative to the horizontal. To draw an elliptic arc, one must again specify `start_angle` and `end_angle`, but these actually do not represent the starting and end angles of the elliptic arc. They are the parameters `t` which parameterize the ellipse `(a*cos(t), b*sin(t))`. This makes drawing elliptic arcs inconvenient.

2. With Tikz in TeX, the position of the arc is specified by where the arc should start drawing. However, it is sometimes easier to specify the *center* of the arc.

With Tikz-Python, `start_angle` and `end_angle` will always coincide with the starting and end angles, so the user will not get weird unexpected behavior. Additionally, the user can specify the arc position via its center by setting `draw_from_start=False`, but they can also fall back on the default behavior.

## Example
Here we draw and fill a sequence of arcs. We also demonstrate `draw_from_start` set to `True` and `False`. In the code below, it is by default set to `True`.
```python
from tikzpy import TikzPicture
from tikzpy.utils import rainbow_colors

tikz = TikzPicture()

for i in range(1, 10):
    t = 4 / i
    arc = tikz.arc((0, 0), 0, 180, radius=t, options=f"fill={rainbow_colors(i)}")

```
This generates the image

<img src="/cs/tikzpy/imgs/arc_ex_1.png"/>

If instead we would like these arcs sharing the same center, we can use the same code, but pass in `draw_from_start=False` to achieve 

<img src="/cs/tikzpy/imgs/arc_ex_2.png"/>

Without this option, if we were forced to specify the point at which each arc should begin drawing, we would have to calculate the x-shift for every arc and apply such a shift to keep the centers aligned. That sounds inefficient and like a waste of time to achieve something so simple, right?


## Methods 
`Arc` has access to methods `.shift()`, `.scale()`, `.rotate()`, which behave as one would expect and takes in parameters as described before.

# `Class: Scope`
Initialize a Tikz scope environment as follows:
```python
from tikzpy import TikzPicture

tikz = TikzPicture()
scope = Scope(options)
```
Or more directly as 
```python
from tikzpy import Scope

scope = Scope(options)
```
As one may guess, `options` is any string of valid Tikz scoping options (i.e., options which become applied to the elements within the scoping evnironment). 

### `Scope.append(draw_obj)`
Appends any drawing object `draw_obj` to the scope environment. If one updates an attribute of a drawing object even after it has been appended, the updates are reflected in scope. 

### `Scope.remove(draw_obj)`
Removes a drawing object `draw_obj` which has been appended to the scoping environment.

### `Scope.clip(draw_obj, draw)`
Clips the drawing object `draw_obj` from the scope environment by creating an instance of the class `Clip`. Here, `draw` is a boolean regarding whether or not you want to actually draw what you are clipping. It is set to `False` by default. 

The class `Scope` also as access to methods `.shift()`, `.scale()`, `.rotate()`. In this case, such operations are applied to every single member of the scoping environment, made possible by the fact that every drawing object itself has access to these methods. These work as one would expect, which is unlike Tikz, since sometimes applying transformations to scoping environments in Tikz does not behave intuitively. 

# `Class: Clip`
A class to clip a single drawing object `draw_obj`.
One can initialize an instance of this class via an instance of `Scope`:
```python
from tikzpy import TikzPicture

tikz = TikzPicture()
scope = tikz.scope()
clip = scope.clip(draw_obj, draw)
```
or more directly as 
```python
from tikzpy import Clip

clip = Clip(draw_obj, draw).
```
As before, `draw` is a boolean set to `False` by default. It specifies whether or not to show the drawing object which is being clipped.

The class `Clip` has access to methods `.shift()`, `.scale()`, `.rotate()`, 
although this is more for consistency (e.g., in case a `Scope` environment changes) and less for direct use of the user. 
