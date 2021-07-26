

from typing import List, Dict, Optional, Tuple, Union, Any, Callable, Optional

import numpy as _numpy

T_ARRAY = _numpy.ndarray

import matplotlib.pyplot as _plt
import matplotlib as _mpl

T_FIG = _mpl.figure.Figure
T_AXE = _mpl.axes._axes.Axes

T_IF = Union[int, float]

def CHECK_NDIM_(arr : T_ARRAY, name : str, ndim : int):

    if arr.ndim != ndim:
        raise ValueError(f"{name}.ndim = {arr.ndim}, must == {ndim}")

def CHECK_IN_(val : Any, name : str, selections : Tuple):

    if val not in selections:
        raise ValueError(f"{name} must in {selections}")