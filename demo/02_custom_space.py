#!/usr/bin/env python3

import numpy as np
from planning_kit import StateSpace


class MyEuclideanSpace:
    def __init__(self, dim):
        self.dim = dim

    def sample_uniform(self):
        return np.random.uniform(0.0, 1.0, self.dim)

    def sample_uniform_near(self, near, dist):
        return near + np.random.uniform(-dist, dist, self.dim)

    def distance(self, a, b):
        return np.linalg.norm(np.subtract(a, b))

    def interpolate(self, a, b, alpha):
        return (1.0 - alpha) * np.asarray(a) + alpha * np.asarray(b)


# Once we wrap our custom implementation in a StateSpace object, we can use it
# as any other.
space = StateSpace.custom(MyEuclideanSpace(dim=3))
