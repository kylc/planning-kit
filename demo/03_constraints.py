#!/usr/bin/env python3

import numpy as np
import rerun as rr
from planning_kit import Constraint, StateSpace


def sinusoid_xy_plane(amp=1.0, freq=1.0):
    def inner(p):
        s = np.sin(2.0 * np.pi * freq * p[0])
        c = np.cos(2.0 * np.pi * freq * p[1])
        z = amp * (s + c)
        return [p[2] - z]

    return inner


constraint = Constraint(3, 1, sinusoid_xy_plane(amp=0.1, freq=0.707))

ambient = StateSpace.euclidean(-np.ones(3), np.ones(3))
projected = StateSpace.projected(ambient, constraint)

# Log everything to Rerun!
rr.init("pk.03_constraints", spawn=True)
rr.log_view_coordinates("space", xyz="FLU", timeless=True)
rr.log_points(
    "space/ambient",
    positions=[ambient.sample_uniform() for _ in range(0, 500)],
    colors=[0.3, 0.3, 0.3],
)
rr.log_points(
    "space/constrained", positions=[projected.sample_uniform() for _ in range(0, 2000)]
)
