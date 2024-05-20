import os

import cv2
import imageio
import numpy as np
import pyrender
import seaborn as sns
import trimesh

from video_mocap.vis.scene import create_floor


def get_point_mesh(
    points,  # [N, 3]
    radius,
    labels=None,  # [N]
    color=None,
):
    color_map = sns.color_palette(palette="YlOrRd", n_colors=10)

    meshes = []
    for i in range(points.shape[0]):
        sphere = trimesh.primitives.Sphere(radius=radius, center=points[i], subdivisions=1)
        if labels is None:
            sphere.visual.vertex_colors = np.array(color)
        else:
            color = color_map[labels[i].item() % 5 + 5]
            sphere.visual.vertex_colors = np.array(color)
        meshes.append(sphere)
    mesh = trimesh.util.concatenate(meshes)
    return mesh


def get_smpl_mesh(
    vertices,
    faces,
):
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
    )
    mesh.visual.vertex_colors = np.array([0.6, 0.6, 0.6, 0.2])
    return mesh


def visualize_part(
    filename,
    markers,  # [F, M, 3]
    vertices,  # [F, V, 3]
    faces,  # [Faces, 3]
    marker_labels,  # [F, M]
    marker_indices,  # [M_p]
    vertex_indices,  # [V_p]
):
    markers_subset = markers[:, marker_indices]
    vertices_subset = vertices[:, vertex_indices]

    num_frames, num_markers, _ = markers_subset.shape
    _, num_vertices, _ = vertices_subset.shape
    marker_size = 0.02
    vertex_size = 0.01

    # create marker meshes
    for _ in range(num_frames):
        sm = trimesh.creation.uv_sphere(radius=marker_size)
        sm.visual.vertex_colors = [1.0, 0.0, 0.0]

        m_tfs = np.tile(np.eye(4), (num_markers, 1, 1))
        m_tfs[:, :3, 3] = markers_subset[0]
        m_mesh = pyrender.Mesh.from_trimesh(sm, poses=m_tfs)

    # create vertices meshes
    for _ in range(num_frames):
        vm = trimesh.creation.uv_sphere(radius=vertex_size)
        vm.visual.vertex_colors = [0.0, 0.1, 0.0]

        v_tfs = np.tile(np.eye(4), (num_vertices, 1, 1))
        v_tfs[:, :3, 3] = vertices_subset[0]
        v_mesh = pyrender.Mesh.from_trimesh(vm, poses=v_tfs)

    # create floor mesh
    f_node = create_floor()

    # render mode
    video_format = os.path.splitext(filename)[1][1:]

    camera_matrix = [
        [0.750, -0.252, 0.611, 3.4],
        [0.661, 0.291, -0.691, -2.0],
        [-0.004, 0.923, 0.385, 2.3],
        [0, 0, 0, 1],
    ]

    scene = pyrender.Scene(ambient_light=[0.2, 0.2, 0.2])

    m_node = scene.add(m_mesh)
    v_node = scene.add(v_mesh)

    f_node = scene.add(f_node)

    # setup lighting
    directional_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
    scene.add(directional_light)

    def render_frame(frame):
        # update markers
        marker_frame = frame

        nonlocal m_node
        nonlocal v_node

        # update markers
        m_tfs[:, :3, 3] = markers_subset[marker_frame]
        m_node.mesh = pyrender.Mesh.from_trimesh(sm, poses=m_tfs)

        # update vertices
        v_tfs[:, :3, 3] = vertices_subset[marker_frame]
        v_node.mesh = pyrender.Mesh.from_trimesh(vm, poses=v_tfs)

    # render output
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    scene.add(camera, pose=camera_matrix)

    renderer = pyrender.OffscreenRenderer(
        viewport_width=640,
        viewport_height=480,
        point_size=1.0,
    )

    flags = pyrender.constants.RenderFlags.SHADOWS_ALL

    if video_format == "ply":
        file_dir = filename.split("." + video_format)[0]
        os.makedirs(file_dir, exist_ok=True)

        for frame in range(0, num_frames, 1):
            render_frame(frame)
            color, _ = renderer.render(scene, flags=flags)
            markers_subset_mesh = get_point_mesh(
                markers_subset[frame],
                radius=0.02,
                labels=marker_labels[frame, marker_indices],
            )
            
            markers_mesh = get_point_mesh(
                markers[frame, np.setdiff1d(np.arange(markers.shape[1]), marker_indices)],
                radius=0.02,
                labels=None,
                color=[0.0, 0.0, 0.0],
            )
            vertices_mesh = get_point_mesh(
                vertices_subset[frame],
                radius=0.01,
                labels=None,
                color=[0.2, 0.6, 1.0],
            )
            smpl_mesh = get_smpl_mesh(
                vertices[frame],
                faces,
            )
            mesh = trimesh.util.concatenate([markers_subset_mesh, markers_mesh, vertices_mesh])#, smpl_mesh])
            mesh.export(os.path.join(file_dir, str(frame) + ".ply"))
            smpl_mesh.export(os.path.join(file_dir, "mesh_" + str(frame) + ".ply"))


    if video_format == "mp4":
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename, fourcc, 30.0, (640, 480))

        # write out video
        for frame in range(0, num_frames, 1):
            render_frame(frame)
            color, _ = renderer.render(scene, flags=flags)
            out.write(np.flip(color, axis=2))

        cap.release()
        out.release()
        cv2.destroyAllWindows()
    elif video_format == "gif":
        # write out video
        with imageio.get_writer(filename, mode="I", duration=(1000.0 * (1.0 / 30.0))) as writer:
            for frame in range(0, num_frames, 1):
                render_frame(frame)
                color, _ = renderer.render(scene, flags=flags)
                writer.append_data(color)

    renderer.delete()
