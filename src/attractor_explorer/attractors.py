"""
Class for working with a family of attractor equations (https://en.wikipedia.org/wiki/Attractor#Strange_attractor)
"""

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import param
import yaml
from numba import jit
from numpy import cos, fabs, sin, sqrt
from param import concrete_descendents

from attractor_explorer.maths import trajectory

RNG = np.random.default_rng(42)


class Attractor(param.Parameterized):
    """Base class for a Parameterized object that can evaluate an attractor trajectory."""

    x = param.Number(0, softbounds=(-2, 2), step=0.01, doc='Starting x value', precedence=-1)
    y = param.Number(0, softbounds=(-2, 2), step=0.01, doc='Starting y value', precedence=-1)
    a = param.Number(1.7, bounds=(-3, 3), step=0.05, doc='Attractor parameter a', precedence=0.2)
    b = param.Number(1.7, bounds=(-3, 3), step=0.05, doc='Attractor parameter b', precedence=0.2)

    colormap: str = 'kgy'
    equations: tuple[str, ...] = ()
    __abstract = True

    @staticmethod
    @jit(cache=True)
    def fn(x, y, a, b, *o):
        pass

    def __call__(self, n: int, x: float | None = None, y: float | None = None) -> pd.DataFrame:
        """Return a dataframe with `n` points."""
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        args = [getattr(self, p) for p in self.signature()]
        return trajectory(self.fn, *args, n=n)

    def update(self, args: dict[str, Any]):
        for key, value in args.items():
            self.__setattr__(key, value)

    def vals(self):
        return [self.__class__.name] + [self.colormap] + [getattr(self, p) for p in self.signature()]

    def signature(self) -> list[str]:
        """Returns the calling signature expected by this attractor function"""
        return list(inspect.signature(self.fn).parameters.keys())[:-1]


class FourParamAttractor(Attractor):
    """Base class for most four-parameter attractors."""

    c = param.Number(0.6, softbounds=(-3, 3), step=0.05, doc='Attractor parameter c', precedence=0.3)
    d = param.Number(1.2, softbounds=(-3, 3), step=0.05, doc='Attractor parameter d', precedence=0.3)
    __abstract = True


class Clifford(FourParamAttractor):
    equations = (r'$x_{n+1} = \sin\ ay_n + c\ \cos\ ax_n$', r'$y_{n+1} = \sin\ bx_n + d\ \cos\ by_n$')

    @staticmethod
    @jit(cache=True)
    def fn(x, y, a, b, c, d, *o):  # noqa: ARG004
        return sin(a * y) + c * cos(a * x), sin(b * x) + d * cos(b * y)


class DeJong(FourParamAttractor):
    equations = (r'$x_{n+1} = \sin\ ay_n - c\ \cos\ bx_n$', r'$y_{n+1} = \sin\ cx_n - d\ \cos\ dy_n$')

    @staticmethod
    @jit(cache=True)
    def fn(x, y, a, b, c, d, *o):  # noqa: ARG004
        return sin(a * y) - cos(b * x), sin(c * x) - cos(d * y)


class Svensson(FourParamAttractor):
    equations = (r'$x_{n+1} = d\ \sin\ ax_n - \sin\ by_n$', r'$y_{n+1} = c\ \cos\ ax_n + \cos\ by_n$')

    @staticmethod
    @jit(cache=True)
    def fn(x, y, a, b, c, d, *o):  # noqa: ARG004
        return d * sin(a * x) - sin(b * y), c * cos(a * x) + cos(b * y)


class FractalDream(FourParamAttractor):
    equations = (r'$x_{n+1} = \sin\ by_n + c\ \sin\ bx_n$', r'$y_{n+1} = \sin\ ax_n + d\ \sin\ ay_n$')

    c = param.Number(1.15, softbounds=(-0.5, 1.5), step=0.05, doc='Attractor parameter c')
    d = param.Number(2.34, softbounds=(-0.5, 1.5), step=0.05, doc='Attractor parameter d')

    @staticmethod
    @jit(cache=True)
    def fn(x, y, a, b, c, d, *o):  # noqa: ARG004
        return sin(b * y) + c * sin(b * x), sin(a * x) + d * sin(a * y)


class Bedhead(Attractor):
    equations = (
        r'$x_{n+1} = y_n\ \sin\ \frac{x_ny_n}{b} + \cos(ax_n-y_n)$',
        r'$y_{n+1} = x_n+\frac{\sin\ y_n}{b}$',
    )

    a = param.Number(0.64, softbounds=(-1, 1))
    b = param.Number(0.76, softbounds=(-1, 1))

    @staticmethod
    @jit(cache=True)
    def fn(x, y, a, b, *o):  # noqa: ARG004
        return y * sin(x * y / b) + cos(a * x - y), x + sin(y) / b

    def __call__(self, n):
        # Avoid interactive divide-by-zero errors for b
        epsilon = 3 * np.finfo(float).eps
        if -epsilon < float(self.b) < epsilon:  # type: ignore
            self.b = float(epsilon)
        return super().__call__(n)


class Hopalong1(Attractor):
    equations = (r'$x_{n+1} = y_n-\mathrm{sgn}(x_n)\sqrt{\left|\ bx_n-c\ \right|}$', r'$y_{n+1} = a-x_n$')

    a = param.Number(9.8, bounds=(0, 10))
    b = param.Number(4.1, bounds=(0, 10))
    c = param.Number(3.8, bounds=(0, 10), doc='Attractor parameter c')

    @staticmethod
    @jit(cache=True)
    def fn(x, y, a, b, c, *o):  # noqa: ARG004
        return y - sqrt(fabs(b * x - c)) * np.sign(x), a - x


class Hopalong2(Hopalong1):
    equations = (
        r'$x_{n+1} = y_n-1-\mathrm{sgn}(x_n-1)\sqrt{\left|\ bx_n-1-c\ \right|}$',
        r'$y_{n+1} = a-x_n-1$',
    )

    @staticmethod
    @jit(cache=True)
    def fn(x, y, a, b, c, *o):  # noqa: ARG004
        return y - 1.0 - sqrt(fabs(b * x - 1.0 - c)) * np.sign(x - 1.0), a - x - 1.0


@jit(cache=True)
def g_func(x, mu):
    return mu * x + 2 * (1 - mu) * x**2 / (1.0 + x**2)


class GumowskiMira(Attractor):
    equations = (
        r'$G(x) = \mu x + \frac{2(1-\mu)x^2}{1+x^2}$',
        r'$x_{n+1} = y_n + ay_n(1-by_n^2) + G(x_n)$',
        r'$y_{n+1} = -x_n + G(x_{n+1})$',
    )
    x = param.Number(0, softbounds=(-20, 20), doc='Starting x value', precedence=0.1)
    y = param.Number(0, softbounds=(-20, 20), doc='Starting y value', precedence=0.1)
    a = param.Number(0.64, softbounds=(-1, 1))
    b = param.Number(0.76, softbounds=(-1, 1))
    mu = param.Number(0.6, softbounds=(-2, 2), step=0.01, doc='Attractor parameter mu', precedence=0.4)

    @staticmethod
    @jit(cache=True)
    def fn(x, y, a, b, mu, *o):  # noqa: ARG004
        xn = y + a * (1 - b * y**2) * y + g_func(x, mu)
        yn = -x + g_func(xn, mu)
        return xn, yn


class SymmetricIcon(Attractor):
    a = param.Number(0.6, softbounds=(-20, 20), step=0.05, doc='Attractor parameter alpha')
    b = param.Number(1.2, softbounds=(-20, 20), step=0.05, doc='Attractor parameter beta')
    gamma = param.Number(0.6, softbounds=(-1, 1), step=0.05, doc='Attractor parameter gamma', precedence=0.5)
    omega = param.Number(1.2, softbounds=(-0.2, 0.2), step=0.01, doc='Attractor parameter omega', precedence=0.5)
    lambda_ = param.Number(0.6, softbounds=(-3, 3), step=0.01, doc='Attractor parameter lambda', precedence=0.5)
    degree = param.Integer(1, softbounds=(1, 25), doc='Attractor parameter degree', precedence=0.5)

    @staticmethod
    @jit(cache=True)
    def fn(x, y, a, b, gamma, omega, lambda_, degree, *o):  # noqa: ARG004
        zzbar = x * x + y * y
        p = a * zzbar + lambda_
        zreal, zimag = x, y

        for _ in range(1, degree - 1):
            za, zb = zreal * x - zimag * y, zimag * x + zreal * y
            zreal, zimag = za, zb

        zn = x * zreal - y * zimag
        p += b * zn

        return p * x + gamma * zreal - omega * y, p * y - gamma * zimag + omega * x


class ParameterSets(param.Parameterized):
    """
    Allows selection from sets of pre-defined parameters saved in YAML.
    """

    data_folder: Path = Path(__file__).parent / 'data'
    input_examples_filename = param.Filename('attractors.yml', search_paths=[data_folder.as_posix()])
    output_examples_filename = param.Filename(
        'saved_attractors.yml', check_exists=False, search_paths=[data_folder.as_posix()]
    )
    current = param.Callable(lambda: None, precedence=-1)
    attractors: dict[str, Attractor] = {}

    load = param.Action(lambda x: x._load())
    randomize = param.Action(lambda x: x._randomize())
    sort = param.Action(lambda x: x._sort())
    remember_this_one = param.Action(lambda x: x._remember())
    # save = param.Action(lambda x: x._save(), precedence=0.8)
    example = param.Selector(objects=[[]], precedence=1, instantiate=False)

    def __init__(self, **params):
        super().__init__(**params)

        self._load()

        self.attractors = {k: v(name=f'{k} parameters') for k, v in sorted(concrete_descendents(Attractor).items())}
        # update attractor instances with the first example of each type
        for attractor in self.attractors:
            try:
                self.get_attractor(attractor, *self.args(attractor)[0])
            except IndexError:
                pass

    def _load(self):
        with Path(self.input_examples_filename).open('r') as f:  # type: ignore
            vals = yaml.safe_load(f)
            if len(vals) > 0:
                self.param.example.objects[:] = vals
                self.example = vals[0]

    # def _save(self):
    #     if self.output_examples_filename == self.param.input_examples_filename.default:
    #         msg = 'Cannot override the default attractors file.'
    #         raise FileExistsError(msg)
    #     with Path(self.data_folder / self.output_examples_filename).open('w') as f:
    #         yaml.dump(self.param.example.objects, f)

    def __call__(self):
        return self.example

    def _randomize(self):
        RNG.shuffle(self.param.example.objects)
        # self.examples = self.param.example.objects
        self.example = self.param.example.objects[0]

    def _sort(self):
        self.param.example.objects[:] = sorted(self.param.example.objects)
        self.example = self.param.example.objects[0]

    def _add_item(self, item):
        self.param.example.objects += [item]
        self.example = item

    def _remember(self):
        vals = self.current().vals()  # type: ignore
        self._add_item(vals)

    def args(self, name):
        return [v[1:] for v in self.param.example.objects if v[0] == name]

    def get_attractor(self, name: str, *args) -> Attractor:
        """Factory function to return an Attractor object with the given name and arg values."""
        attractor = self.attractors[name]
        fn_params = ['colormap', *attractor.signature()]
        # attractor.param.update(**dict(zip(fn_params, args, strict=True)))
        attractor.update(dict(zip(fn_params, args, strict=True)))
        return attractor
