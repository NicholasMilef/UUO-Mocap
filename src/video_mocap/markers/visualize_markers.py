import argparse
import time

import trimesh
import pyrender
import numpy as np

from video_mocap.markers.markers import Markers


def visualize_markers(filename):
    markers = Markers(filename)
    points = markers.get_points() / 100
    points = np.nan_to_num(points, 0)
    freq = markers.get_frequency()

    # create marker meshes
    sm = trimesh.creation.uv_sphere(radius=0.1)
    sm.visual.vertex_colors = [1.0, 0.0, 0.0]
    tfs = np.tile(np.eye(4), (points.shape[1], 1, 1))
    tfs[:, :3, 3] = points[0]
    m = pyrender.Mesh.from_trimesh(sm, poses=tfs)

    # create floor mesh
    floor_mesh = trimesh.Trimesh(
        vertices=[[-25, -25, 0], [-25, 25, 0], [25, -25, 0], [25, 25, 0]],
        faces=[[0, 2, 1], [1, 2, 3]],
    )
    f_node = pyrender.Mesh.from_trimesh(floor_mesh)

    # setup scene
    scene = pyrender.Scene()
    m_node = scene.add(m)
    f_node = scene.add(f_node)
    viewer = pyrender.Viewer(
        scene,
        use_raymond_lighting=False,
        run_in_thread=True,
        shadows=True,
    )

    # setup lighting
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
    scene.add(light)

    frame = 0
    while viewer.is_active:
        viewer.render_lock.acquire()

        tfs[:, :3, 3] = points[frame]
        m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        m_node.mesh = m

        viewer.render_lock.release()

        frame = (frame + 1) % points.shape[0]
        time.sleep(1.0 / freq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, help="filename for .c3d file", required=True)
    args = parser.parse_args()

    markers = visualize_markers(args.filename)
