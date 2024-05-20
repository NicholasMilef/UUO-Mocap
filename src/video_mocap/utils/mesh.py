from copy import deepcopy

import numpy as np
from trimesh import Trimesh


def cull_parts(
    mesh: Trimesh,
    parts: np.ndarray,  # [J_markers]
    vertex_labels: np.ndarray,  # [V]
) -> Trimesh:
    """
    Cull parts of the mesh based on a parts mask

    Args:
        mesh: original Trimesh
        parts: labels of parts that should be kept [J_markers]
        vertex_labels: labels of each vertex to part

    Returns:
        Trimesh: updated mesh with only vertices corresponding to parts
    """
    vertex_indices = {}
    for part_id in parts:
        part_indices = (vertex_labels == part_id).nonzero()[0]
        for part_index in part_indices:
            vertex_indices[part_index] = True

    #vertex_indices = np.concatenate(vertex_indices, axis=0)

    face_mask = np.zeros(mesh.faces.shape[0], dtype=bool)
    face_index = 0
    for face in mesh.faces:
        if face[0] in vertex_indices or face[1] in vertex_indices or face[2] in vertex_indices:
            face_mask[face_index] = True
        face_index += 1

    output_mesh = deepcopy(mesh)
    output_mesh.update_faces(face_mask)

    return output_mesh
