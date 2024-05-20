import igl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh

from video_mocap.utils.smpl import SmplInferenceGender


def create_smpl_sdf():
    device = torch.device("cpu")
    bm = SmplInferenceGender(device)

    output = bm(
        poses=torch.zeros((1, 1, 69)).to(device),
        betas=torch.zeros((1, 10)).to(device),
        root_orient=torch.zeros((1, 1, 3)).to(device),
        trans=torch.zeros((1, 1, 3)).to(device),
        gender_one_hot=torch.FloatTensor([[1.0, 0.0]]).to(device),
        pose2rot=True,
    )
    vertices = output["vertices"].detach().cpu().numpy()
    faces = bm.smpls["male"].faces

    padding = 0.1  # padding for SDF
    res = (512, 512, 128)

    min_x = np.min(vertices[0, 0, :, 0]) - padding
    max_x = np.max(vertices[0, 0, :, 0]) + padding
    min_y = np.min(vertices[0, 0, :, 1]) - padding
    max_y = np.max(vertices[0, 0, :, 1]) + padding
    min_z = np.min(vertices[0, 0, :, 2]) - padding
    max_z = np.max(vertices[0, 0, :, 2]) + padding

    samples = np.mgrid[0:res[0], 0:res[1], 0:res[2]].astype(np.float32)  # [3, X, Y, Z]
    samples = samples.transpose((1, 2, 3, 0))  # [X, Y, Z, 3]
    samples[..., 0] = samples[..., 0] / (res[0] - 1.0)
    samples[..., 1] = samples[..., 1] / (res[1] - 1.0)
    samples[..., 2] = samples[..., 2] / (res[2] - 1.0)

    # scale x, y, z
    samples[..., 0] = (samples[..., 0] * (max_x - min_x)) + min_x
    samples[..., 1] = (samples[..., 1] * (max_y - min_y)) + min_y
    samples[..., 2] = (samples[..., 2] * (max_z - min_z)) + min_z
        
    body_mesh = trimesh.Trimesh(
        vertices=vertices[0, 0],
        faces=faces,
        process=False,
    )

    samples = np.reshape(samples, (-1, 3))
    distances, face_indices, points = igl.signed_distance(samples, body_mesh.vertices, body_mesh.faces)

    i0 = body_mesh.faces[face_indices][:, 0]  # [N]
    i1 = body_mesh.faces[face_indices][:, 1]  # [N]
    i2 = body_mesh.faces[face_indices][:, 2]  # [N]

    v0 = body_mesh.vertices[i0]  # [N, 3]
    v1 = body_mesh.vertices[i1]  # [N, 3]
    v2 = body_mesh.vertices[i2]  # [N, 3]

    triangles = np.stack((v0, v1, v2), axis=1)

    barycentric_coords = trimesh.triangles.points_to_barycentric(
        triangles,
        points,
    )  # [N, 3]

    output = {}
    output["barycentric_coords"] = barycentric_coords.reshape((res[0], res[1], res[2], 3))
    output["points"] = points.reshape((res[0], res[1], res[2], 3))
    output["samples"] = samples.reshape((res[0], res[1], res[2], 3))
    output["signed_distances"] = distances.reshape((res[0], res[1], res[2]))
    return output


def normalize_channel(points):
    min_value = np.min(points)
    max_value = np.max(points)
    return (points - min_value) / (max_value - min_value)


def visualize_sdf():
    sdf = create_smpl_sdf()

    signed_distances = sdf["signed_distances"]
    signed_distances[signed_distances < 0] = 0
    signed_distances[signed_distances > 0] = 1

    points = sdf["points"]
    points[..., 0] = normalize_channel(points[..., 0]) * signed_distances
    points[..., 1] = normalize_channel(points[..., 1]) * signed_distances
    points[..., 2] = normalize_channel(points[..., 2]) * signed_distances

    fig = plt.figure()
    im = plt.imshow(points[:, :, 0])

    def anim_func(i):
        im.set_array(points[:, :, i])
        return [im]

    anim = animation.FuncAnimation(
        fig,
        anim_func,
        frames=points.shape[2],
        interval=1000.0/60.0,
    )
    anim.save("./data/sdf.gif")

    for key, value in sdf.items():
        sdf[key] = sdf[key].astype(np.float16)

    np.savez("smpl_sdf", **sdf)

    plt.show()



if __name__ == "__main__":
    visualize_sdf()

