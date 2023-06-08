#!/usr/bin/env python3

import io
import os
from typing import Dict, Optional, cast
from urllib.parse import urlparse

import numpy as np
import rerun as rr
import trimesh

# from ament_index_python.packages import get_package_share_directory
# from std_msgs.msg import String
from yourdfpy import URDF


def strip_package_prefix(fname: str) -> str:
    if not fname.startswith("package://"):
        return fname
    parsed = urlparse(fname)
    return parsed.path[1:]


def load_urdf_from_file(model_path) -> URDF:
    """Load a URDF file using `yourdfpy` and find resources."""
    with open(model_path, "rb") as f:
        return URDF.load(f, filename_handler=strip_package_prefix)


def log_scene(
    scene: trimesh.Scene, node: str, path: Optional[str] = None, timeless: bool = False
) -> None:
    """Log a trimesh scene to rerun."""
    path = path + "/" + node if path else node

    parent = scene.graph.transforms.parents.get(node)
    children = scene.graph.transforms.children.get(node)

    node_data = scene.graph.get(frame_to=node, frame_from=parent)

    if node_data:
        # Log the transform between this node and its direct parent (if it has one!).
        if parent:
            world_from_mesh = node_data[0]
            rr.log_transform3d(
                path,
                rr.TranslationAndMat3(
                    translation=trimesh.transformations.translation_from_matrix(
                        world_from_mesh
                    ),
                    matrix=world_from_mesh[0:3, 0:3],
                ),
                timeless=timeless,
            )

        # Log this node's mesh, if it has one.
        mesh = cast(trimesh.Trimesh, scene.geometry.get(node_data[1]))
        if mesh:
            # If vertex colors are set, use the average color as the albedo factor
            # for the whole mesh.
            vertex_colors = None
            try:
                colors = np.mean(mesh.visual.vertex_colors, axis=0)
                if len(colors) == 4:
                    vertex_colors = np.array(colors) / 255.0
            except Exception:
                pass

            # If trimesh gives us a single vertex color for the entire mesh, we can interpret that
            # as an albedo factor for the whole primitive.
            visual_color = None
            try:
                colors = mesh.visual.to_color().vertex_colors
                if len(colors) == 4:
                    visual_color = np.array(colors) / 255.0
            except Exception:
                pass

            albedo_factor = vertex_colors if vertex_colors is not None else visual_color

            rr.log_mesh(
                path,
                mesh.vertices,
                indices=mesh.faces,
                normals=mesh.vertex_normals,
                albedo_factor=albedo_factor,
                timeless=timeless,
            )

    if children:
        for child in children:
            log_scene(scene, child, path, timeless)


def log_scene_state(
    scene: trimesh.Scene,
    transforms: Dict[str, np.array],
    node: str,
    tfm: np.array = np.eye(4),
    path: Optional[str] = None,
    timeless: bool = False,
) -> None:
    path = path + "/" + node if path else node

    parent = scene.graph.transforms.parents.get(node)
    children = scene.graph.transforms.children.get(node)

    if node in transforms:
        oMp = tfm
        oMc = transforms[node]

        pMc = np.linalg.pinv(oMp) @ oMc

        rr.log_transform3d(
            path,
            rr.TranslationAndMat3(translation=pMc[0:3, 3], matrix=pMc[0:3, 0:3]),
            from_parent=False,
            timeless=timeless,
        )

        if children:
            for child in children:
                log_scene_state(scene, transforms, child, oMc, path, timeless)


def log_arm_problem_scene(model, collision_mesh, problem, graph, pathspl, ee_frame):
    log_scene(model.scene, model.base_link, path="robot", timeless=True)
    rr.log_mesh(
        "robot/collision",
        positions=collision_mesh.vertices,
        indices=collision_mesh.indices,
        albedo_factor=[0.2, 0.2, 0.2],
        timeless=True,
    )

    fks = [problem.forward_kinematics(q) for q in graph.points()]
    rr.log_points(
        "robot/rrt/graph", positions=[fk[ee_frame][0:3, 3] for fk in fks], timeless=True
    )

    fk_interp = []
    for t in np.linspace(0.0, 1.0, num=1000):
        rr.set_time_seconds("interp_time", t)
        q = pathspl(t)
        fk = problem.forward_kinematics(q)
        fk_interp.append(fk[ee_frame][0:3, 3])
        log_scene_state(model.scene, node=model.base_link, path="robot", transforms=fk)

    rr.log_line_strip("robot/rrt/path", positions=fk_interp, timeless=True)
