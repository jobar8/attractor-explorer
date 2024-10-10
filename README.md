# Attractor Explorer

<!-- <img src="https://raw.githubusercontent.com/jobar8/attractors2023/master/docs/source/_static/assets/images/panel_screenshot.png" alt="Attractors Panel" width="800" role="img"> -->

<br>

[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-360/)
<!-- [![CI - Test](https://github.com/jobar8/attractor-explorer/actions/workflows/checks.yml/badge.svg)](https://github.com/jobar8/attractor-explorer/actions/workflows/checks.yml)  -->

-----

**Table of Contents**

- [Attractor Explorer](#attractor-explorer)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Code](#code)
  - [License](#license)

## Overview

I have always been fascinated by [patterns](https://en.wikipedia.org/wiki/Pattern), [fractals](https://en.wikipedia.org/wiki/Fractal)
and [attractors](https://en.wikipedia.org/wiki/Attractor).
So when attractors display a particular pattern with a fractal structure, the so-called [strange attractors](https://en.wikipedia.org/wiki/Attractor#Strange_attractor), I tend to be even more *attracted* to them. 

It turns out I am not the only one interested in those strange mathematical objects and one of the most effective
ways to explore various types of attractors and their parameters
is to use the `attractors_panel` [interactive dashboard](https://examples.holoviz.org/gallery/attractors/attractors_panel.html) developed a
few years ago by the nice people of [HoloViz](https://holoviz.org/). The dashboard is powered
by [panel](https://panel.holoviz.org/) and makes heavy use of [numba](https://numba.pydata.org/)
and [Datashader](https://datashader.org) to speed up calculations and rendering.

This project, *Attractor Explorer*, has two objectives:
- to provide an updated version of the original dashboard
- to provide a new dashboard that allows users to animate the plots by automatically varying parameters within a certain range

My contribution has first consisted in re-organising the original code into a proper Python package, with unit tests
and documentation. 

## Installation

This project is still work in progress and has not been released as a distributed package yet. So it is
necessary to first clone the Git repo before installing the library.

First create a virtual environment (local or not) using your favourite environment manager (hatch, poetry, conda).
Once this is done, simply install the package (in editable mode or not) with:

```sh
pip install -e .
```

Alternatively, you can also create a local environment and install all the dependencies by using
[uv](https://docs.astral.sh/uv/):

```sh
uv run panel serve src/attractor_explorer/attractors_explorer.py
```

This would also start the dashboard (see below).


## Usage

Two `panel` webapps or "dashboards" are available. The first one is a modified version of `attractors_panel` and
it can be launched with:

```sh
panel serve src/attractor_explorer/attractors_explorer.py
```

It should work similarly to the original one: select the type of attractor, the resolution (the number of points
calculated), and then use the cursors to play with the parameters.


## Code

Each attractor has:

- Executable Python code for calculating trajectories, optimized using [Numba](https://numba.pydata.org).
- Readable equations displayable with KaTeX
- Examples of interesting patterns stored in a separate `attractors.yml` file

Support is provided for reading the `attractors.yml` file and working with the examples in it.

The changes I have introduced compared to the original HoloViz version include:

- linting and formatting with [Ruff](https://docs.astral.sh/ruff)
- data folder for storing parameters
- renaming of attractor classes to follow CapWords convention (`Fractal_Dream` -> `FractalDream`)
- use `item_type` instead of `class_` attribute in `param.List()`
- rename `.sig()` method `.signature()`
- use `np.random.default_rng()` instead of `numpy.random.seed()`
- move `trajectory_coords()` and `trajectory()` functions to new `maths.py` module
- unit tests and Github actions

## License

`attractor-explorer` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
