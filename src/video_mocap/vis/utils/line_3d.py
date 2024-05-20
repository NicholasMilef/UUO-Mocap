import numpy as np
import trimesh


class Line3D:
    def __init__(self, radius=0.01):
        self.mesh = trimesh.creation.capsule(
            radius=radius,
            height=1.0,
        )

    def get_mesh(self):
        return self.mesh

    @staticmethod
    def get_matrix(pos_0, pos_1):
        midpoint = (pos_0 + pos_1) / 2
        tfs = np.eye((4))[None]
        
        # apply rotation
        right = np.array([1, 0, 0])
        up = (pos_0 - pos_1) / np.linalg.norm(pos_0 - pos_1)
        forward = np.cross(up, right)
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        tfs[:, :3, :3] = np.transpose(np.stack((right, forward, up), axis=0))

        # apply translation
        tfs[:, :3, 3] = midpoint
        
        # apply scale
        scale_matrix = np.eye((4))[None]
        scale_matrix[:, 2, 2] = np.linalg.norm(pos_0 - pos_1)
        tfs = tfs @ scale_matrix

        return tfs