# Kabak
Algorithms for covering and packing problems.

_Kabak_ is a phonetic pormantau of _cover_ and _pack_. It means a pub in Russian.

## Overview
This package implements algorithms for covering and packing problems. The main goal is to implement approximation algorithms, heuristics, and other structure-exploiting special algorithms that are not available elsewhere.

## Covering and Packing
Covering / Packing problems are linear programming / integer programming problems with non-negative inputs. A general covering problem has the form

$$
\begin{align}
\min c^T &x \\
\text{s.t. } A&x \geq b \\
    B&x \leq f \\
     &x \leq d\\
     x \geq 0
\end{align}
$$


## Contributing
There are very many packing and covering models. Help with implementing more of them is much appreciated.

New applications may also be featured.


### Documentation
`kabak` uses [Sphinx-rtd](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/index.html) for documentation. Ideally functions are type-annotated according to [sphinx-autodoc-typehints](https://pypi.org/project/sphinx-autodoc-typehints/).

There is more reading on documentation at e.g. [sphinx-sublime-gitHub](https://sublime-and-sphinx-guide.readthedocs.io/en/latest/index.html). See also the [sphinx style guide](https://documentation-style-guide-sphinx.readthedocs.io/en/latest/contribute/index.html).

#### Mathematical notation in the documentation
We use `sphinx.ext.imgmath`. 

---
***NOTE***
Keep in mind that when you put math markup in Python docstrings read by autodoc, you either have to double all backslashes, or use Python raw strings (``r"raw"``).

---
