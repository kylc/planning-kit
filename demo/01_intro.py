#!/usr/bin/env python3

from math import sqrt

import numpy as np
import planning_kit as pkit
from planning_kit import StateSpace, Constraint

# We can construct a Euclidean space, from which samples will be drawn. Because
# this space is defined in the native Rust code, sampling is computationally
# efficient.
#
# We define lower and upper boundaries of the space in R^3.
euclidean = StateSpace.euclidean(-np.ones(3), np.ones(3))


# A sample is valid if it lies outside a sphere of radius 0.95.
def is_valid(s):
    return sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 3) > 0.95


# Now we can use a variety of the built-in sampling-based motion planning
# algorithms.
tree = pkit.rrt_connect(
    euclidean,  # the search space
    is_valid,  # is a particular sample valid?
    start=-np.ones(3),  # the start state
    goal=np.ones(3),  # the goal state
    discretization=0.01,  # motion validation resolution
    # planner-specific parameters
    steering_dist=0.1,
    n=10000,
)
