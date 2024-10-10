# Attractor Explorer

![Attractor Explorer Screenshot](docs/source/_static/assets/images/explorer_screenshot.png)

<br>

[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-360/)
![Linting - Tests](https://github.com/jobar8/attractor-explorer/actions/workflows/checks.yml/badge.svg)

-----

**Table of Contents**

- [Attractor Explorer](#attractor-explorer)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Usage](#usage)
  - [License](#license)

## Overview

I have always been fascinated by [patterns](https://en.wikipedia.org/wiki/Pattern), [fractals](https://en.wikipedia.org/wiki/Fractal)
and [attractors](https://en.wikipedia.org/wiki/Attractor).
So when attractors display a particular pattern with a fractal structure, the so-called [strange attractors](https://en.wikipedia.org/wiki/Attractor#Strange_attractor),
I tend to be even more *attracted* to them. 

It turns out I am not the only one interested in those strange mathematical objects and one of the most effective ways to explore various types of attractors and their parameters
is to use the [interactive webapp](https://attractors.pyviz.demo.anaconda.com/attractors_panel) developed a few years ago by the nice people of Anaconda. The dashboard is powered
by [panel](https://panel.holoviz.org/) and makes heavy use of [numba](https://numba.pydata.org/) and [Datashader](https://datashader.org) to speed up calculations and rendering.

This project, *Attractors 2023*, has two objectives:
- to provide an updated version of the original dashboard
- to provide a new dashboard that can allow the user to *zoom* into the attractor's trajectory

My contribution has first consisted in re-organising the original code into a proper Python package, with unit tests and documentation. I then added a new method for
calculating the trajectory (the set of points formed by the imaginary movement of a particle according to the attractor's equations). The method uses multiprocessing to calculate
the trajectory from several initial points in parallel.

## Installation

This project is still work in progress and has not been released as a distributed package yet.

To run and test the code, the easiest is to install the dependencies in a virtual environment using your favourite environment manager.
Once this is done, simply install the package in editable mode with:

```console
pip install -e .
```

Alternatively, since the `pyproject.toml` has been configured with [hatch](https://hatch.pypa.io/latest/), a local environment can be created by running:

```
hatch env create
```

You can then enter the environment with:

```
hatch shell
```

Note: The dependencies are not compatible with Python 3.12.

## Usage

Two `panel` webapps or "dashboards" are available. The first one is the updated version of `attractors_panel` and it can be launched with:

```console
panel serve --show src/attractors2023/attractors_panel.py
```

It should work just like the original one.

The second dashboard is very similar, except for the presence of the two X and Y coordinates sliders (and the absence of the "player"). Run it with:

```console
panel serve --show src/attractors2023/attractors_explorer.py
```

The X and Y sliders allow you to limit the display of the attractor to a particular area. 


## License

`attractor-explorer` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
