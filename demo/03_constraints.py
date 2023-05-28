#!/usr/bin/env python3

import numpy as np
from planning_kit import Constraint, StateSpace


def sphere(s):
    return [s[0] ** 2 + s[1] ** 2 + s[2] ** 2 - 1.0]


sphere_constraint = Constraint(3, 1, sphere)

ambient = StateSpace.euclidean(-np.ones(3), np.ones(3))
projected = StateSpace.projected(ambient, sphere_constraint)
