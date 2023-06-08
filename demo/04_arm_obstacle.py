#!/usr/bin/env python3

import sys

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline
import rerun as rr
import rerun_ext as rr_ext
import trimesh

import planning_kit as pkit


def load_mesh(path):
    """Helper function to build a planning_kit.Mesh object using the trimesh
    library's loading functionality."""
    mesh = trimesh.load_mesh(path)
    vertices = np.asarray(mesh.vertices).flatten()
    indices = np.asarray(mesh.faces).flatten()
    return pkit.Mesh(vertices, indices)


urdf_model_path = sys.argv[1]
collision_mesh_path = sys.argv[2]
ee_frame = sys.argv[3]

with open(urdf_model_path, "r") as f:
    urdf = f.read()

collision_mesh = load_mesh(collision_mesh_path)

# Build a pre-baked problem definition for soliving a collision-free joint-space
# trajectory of a kinematic chain.
arm_problem = pkit.KinematicChainProblem(urdf, collision_mesh)

# Define our search space.
nq = arm_problem.nq()
space = pkit.StateSpace.euclidean(-3.14 * np.ones(nq), 3.14 * np.ones(nq))

# Define our start and goal configurations as joint angles.
q_init = np.zeros(nq)
q_goal = np.zeros(nq)
q_goal[0] = 3.0

# Perform an RRT-Connect search. This can easily be replaced with a different
# planning algorithm because the specifics of the search space and state
# validation are abstracted.
graph = pkit.rrt_connect_problem(
    space,
    arm_problem,
    start=q_init,
    goal=q_goal,
    discretization=0.01,
    steering_dist=0.1,
)

# Find the points along the shortest path.
path = graph.shortest_path(space, q_init, q_goal)

# Perform a naive joint-space B-spline interpolation to get a smooth path.
splx = np.linspace(0.0, 1.0, num=len(path.points()))
spl = make_interp_spline(splx, path.points())

# Log everything to Rerun!
rr.init("pk.04_arm_obstacle", spawn=True)
rr.log_view_coordinates("robot", xyz="FLU", timeless=True)
viz_mod = rr_ext.load_urdf_from_file(urdf_model_path)
rr_ext.log_arm_problem_scene(viz_mod, collision_mesh, arm_problem, graph, spl, ee_frame)
