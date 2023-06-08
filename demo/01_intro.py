#!/usr/bin/env python3

from math import sqrt

import numpy as np
import rerun as rr
import planning_kit as pkit
from planning_kit import StateSpace, Constraint

# We can construct a Euclidean space, from which samples will be drawn. Because
# this space is defined in the native Rust code, sampling is computationally
# efficient.
#
# We define lower and upper boundaries of the space in R^3.
euclidean = StateSpace.euclidean(-np.ones(3), np.ones(3))

R = 0.95

# A sample is valid if it lies outside a sphere of radius 0.95.
def is_valid(s):
    return sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 3) > R


# Now we can use a variety of the built-in sampling-based motion planning
# algorithms.
start = -np.ones(3)
goal = np.ones(3)
tree = pkit.rrt_connect(
    euclidean,  # the search space
    is_valid,  # is a particular sample valid?
    start=start,  # the start state
    goal=goal,  # the goal state
    discretization=0.01,  # motion validation resolution
    # planner-specific parameters
    steering_dist=0.1,
    n=10000,
)

# Log everything to Rerun!
rr.init("pk.01_intro", spawn=True)
rr.log_view_coordinates("space", xyz="FLU", timeless=True)
rr.log_points("space/rrt_tree", positions=[p for p in tree.points()])
rr.log_line_strip(
    "space/shortest_path",
    positions=tree.shortest_path(euclidean, start, goal).points(),
)

# Sample the surface of the sphere so we can visualize the keep-out region.
xs = np.random.uniform(low=-1.0, high=1.0, size=(1000, 3))
xs = R * xs / np.linalg.norm(xs, axis=1)[:, None]
rr.log_points("space/sphere", positions=xs)
