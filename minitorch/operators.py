"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -x


def lt(x: float, y: float) -> float:
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    # if isinstance(x, tuple):
    #     x = x[0]
    return 1.0 / (1.0 + math.exp(-x)) if x > 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    return x if x > 0 else 0


def log(x: float) -> float:
    return math.log(x)


def exp(x: float) -> float:
    # if isinstance(x, tuple):
    #     x = x[0]
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    return d * 1 / (x)


def inv(x: float) -> float:
    return 1 / x


def inv_back(x: float, d: float) -> float:
    # If x is a tuple, unpack it
    # if isinstance(x, tuple):
    #     x = x[0]
    return d * (-1 / x**2)


def relu_back(x: float, d: float) -> float:
    # if isinstance(x, tuple):
    #     x = x[0]
    return d if x > 0 else 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.


# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
    ----
        fn: Function from one value to one value.

    Returns:
    -------
        A function that takes a list, applies `fn` to each element, and returns a
         new list

    """
    # TODO: Implement for Task 0.3.

    def map_fun(list1: Iterable[float]) -> Iterable[float]:
        new_list = []
        for v in list1:
            new_list.append(fn(v))
        return new_list

    return map_fun


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Use `map` and `neg` to negate each element in `ls`"""
    # TODO: Implement for Task 0.3.
    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
    ----
        fn: combine two values

    Returns:
    -------
        Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """

    # TODO: Implement for Task 0.3.
    def zip_function(list1: Iterable[float], list2: Iterable[float]) -> Iterable[float]:
        new_list = []
        for i, j in zip(list1, list2):
            new_list.append(fn(i, j))
        return new_list

    return zip_function


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add the elements of `ls1` and `ls2` using `zipWith` and `add`"""
    # TODO: Implement for Task 0.3.
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""Higher-order reduce.

    Args:
    ----
        fn: combine two values
        start: start value $x_0$

    Returns:
    -------
        Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`

    """

    # TODO: Implement for Task 0.3.
    def reduce_function(ls: Iterable[float]) -> float:
        result = start
        for i in ls:
            result = fn(result, i)
        return result

    return reduce_function


def sum(ls: Iterable[float]) -> float:
    """Sum up a list using `reduce` and `add`."""
    # TODO: Implement for Task 0.3.
    return reduce(add, 0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Product of a list using `reduce` and `mul`."""
    # TODO: Implement for Task 0.3.
    return reduce(mul, 1)(ls)
