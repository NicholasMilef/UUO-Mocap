from collections.abc import Callable
import math
from typing import Dict

import igl
import numpy as np
from pytorch3d.loss.chamfer import chamfer_distance
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d, so3_relative_angle, axis_angle_to_matrix
import torch
import torch.nn.functional as F
import trimesh

from video_mocap.losses.chamfer_distance import weighted_chamfer_distance
from video_mocap.losses.losses import MarkerLoss
from video_mocap.utils.points import geometric_median, closest_point
from video_mocap.utils.sdf import SDF
from video_mocap.utils.settings import MARKER_DISTANCE
from video_mocap.utils.smpl import SmplInference


def optim_root(
    markers: torch.Tensor,  # [F, M, 3]
    pose_body: torch.Tensor,  # [F, J-1, 3, 3]
    betas: torch.Tensor,  # [F, 10]
    root_orient: torch.Tensor,  # [F, 1, 3, 3]
    trans: torch.Tensor,  # [F, 3]
    marker_labels: torch.Tensor,  # [F, M]
    smpl_inference: SmplInference,
    config: Dict,
    verbose: bool=False,
    iter_fn: Callable=None,
):
    # optimization setup
    if config["stages"]["root"]["constrained_rotation"]:
        z_angle = torch.zeros((1, root_orient.shape[1], 1)).to(root_orient.device).requires_grad_(True)
    elif config["stages"]["root"]["yaw_lock"]:
        z_angle = torch.zeros((root_orient.shape[0], root_orient.shape[1], 1)).to(root_orient.device).requires_grad_(True)
    else:
        z_angle = torch.zeros((root_orient.shape[0], root_orient.shape[1], 3, 3)).to(root_orient.device)
        z_angle[..., 0, 0] = 1.0
        z_angle[..., 1, 1] = 1.0
        z_angle[..., 2, 2] = 1.0
        z_angle.requires_grad_(True)
    params = [trans] + [z_angle] + [betas]

    optimizer = torch.optim.LBFGS(
        params,
        max_iter=config["stages"]["root"]["num_iters"],
        tolerance_grad=config["optimizer"]["tolerance_grad"],
        tolerance_change=config["optimizer"]["tolerance_change"],
        lr=config["stages"]["root"]["lr"],
        line_search_fn="strong_wolfe",
    )
    root_orient.requires_grad_(False)

    iteration = 0
    def closure_stage_translation():
        optimizer.zero_grad()
        nonlocal iteration

        if config["stages"]["root"]["constrained_rotation"]:
            z_root_orient = compute_root_orient_z(torch.repeat_interleave(z_angle, repeats=root_orient.shape[0], dim=0)) @ root_orient
        elif config["stages"]["root"]["yaw_lock"]:
            z_root_orient = compute_root_orient_z(z_angle) @ root_orient
        else:
            z_root_orient = rotation_6d_to_matrix(matrix_to_rotation_6d(z_angle))

        root_orient_vel = so3_relative_angle(
            rotation_6d_to_matrix(matrix_to_rotation_6d(root_orient[1:, 0])),
            rotation_6d_to_matrix(matrix_to_rotation_6d(root_orient[:-1, 0])),
        )
        z_root_orient_vel = so3_relative_angle(
            rotation_6d_to_matrix(matrix_to_rotation_6d(z_root_orient[1:, 0])),
            rotation_6d_to_matrix(matrix_to_rotation_6d(z_root_orient[:-1, 0])),
        )

        smpl_output = smpl_inference(
            poses=pose_body,
            betas=torch.repeat_interleave(betas, dim=0, repeats=pose_body.shape[0]),
            root_orient=z_root_orient,
            trans=trans,
        )

        #loss, _ = chamfer_distance(markers, smpl_output["vertices"], single_directional=True)

        loss = 0
        if "part_chamfer" in config["stages"]["root"]["losses"]:
            loss_chamfer = chamfer_distance_by_part(
                markers,
                smpl_output["vertices"],
                marker_labels,
                smpl_inference.smpl.lbs_weights,
                single_directional=config["stages"]["root"]["single_directional"],
            )
            loss += loss_chamfer * config["stages"]["root"]["losses"]["part_chamfer"]
        if "full_chamfer" in config["stages"]["root"]["losses"]:
            loss_chamfer = weighted_chamfer_distance(
                x=markers,
                y=smpl_output["vertices"],
                x_weights=get_marker_mask(markers),
                single_directional=config["stages"]["root"]["single_directional"],
            )[0]
            loss += loss_chamfer * config["stages"]["root"]["losses"]["full_chamfer"]
        if "root_orient_vel" in config["stages"]["root"]["losses"]:
            loss += F.mse_loss(z_root_orient_vel, root_orient_vel) * config["stages"]["root"]["losses"]["root_orient_vel"]
        if "trans_vel" in config["stages"]["root"]["losses"]:
            trans_vel = trans[1:] - trans[:-1]  # [F-1, 3]
            markers_mean = torch.mean(markers, dim=1)  # [F, 3]
            markers_vel = markers_mean[1:] - markers_mean[:-1]  # [F-1, 3]
            loss += F.mse_loss(trans_vel, markers_vel) * config["stages"]["root"]["losses"]["trans_vel"]
        if "reg_betas" in config["stages"]["root"]["losses"]:
            loss += F.mse_loss(betas, o_betas) * config["stages"]["root"]["losses"]["reg_betas"]
        if "ground" in config["stages"]["root"]["losses"]:
            import pdb; pdb.set_trace()
            ground_loss = torch.mean(F.relu(-smpl_output["joints"][..., 2]))
            #ground_loss = torch.mean(F.relu(-torch.min(smpl_output["joints"][..., 2])))
            loss += ground_loss * config["stages"]["root"]["losses"]["ground"]

        loss.backward()

        if verbose:
            print("Root", iteration, float(loss))
        if iter_fn is not None:
            iter_fn(
                stage="root",
                iteration=iteration,
                pose_body=pose_body.detach().cpu().numpy(),
                betas=betas.detach().cpu().numpy(),
                trans=trans.detach().cpu().numpy(),
                root_orient=rotation_6d_to_matrix(matrix_to_rotation_6d(z_root_orient)).detach().cpu().numpy(),
            )

        iteration += 1
        return loss

    optimizer.step(closure_stage_translation)
    if config["stages"]["root"]["constrained_rotation"]:
        root_orient[:] = compute_root_orient_z(torch.repeat_interleave(z_angle, repeats=root_orient.shape[0], dim=0)) @ root_orient
    elif config["stages"]["root"]["yaw_lock"]:
        root_orient[:] = compute_root_orient_z(z_angle) @ root_orient
    else:
        root_orient[:] = rotation_6d_to_matrix(matrix_to_rotation_6d(z_angle)) @ root_orient
    root_orient.detach()
    root_orient.requires_grad_(True)


def optim_chamfer(
    markers: torch.Tensor,  # [F, M, 3]
    pose_body: torch.Tensor,  # [F, J-1, 3, 3]
    o_pose_body: torch.Tensor,  # [F, J-1, 3, 3]
    betas: torch.Tensor,  # [F, 10]
    o_betas: torch.Tensor,  # [F, 10]
    root_orient: torch.Tensor,  # [F, 1, 3, 3]
    trans: torch.Tensor,  # [F, 3]
    img_mask: torch.Tensor,  # [F]
    marker_labels: torch.Tensor,  # [F, M]
    smpl_inference: SmplInference,
    config: Dict,
    initial_angle: float=0,
    repeat: int=0,
    verbose: bool=False,
    iter_fn: Callable=None,
):
    # optimization setup
    if config["stages"]["chamfer"]["yaw_lock"]:
        z_angle = torch.zeros((root_orient.shape[0], root_orient.shape[1], 1)).to(root_orient.device).requires_grad_(True)
    else:
        z_angle = torch.zeros((root_orient.shape[0], root_orient.shape[1], 3, 3)).to(root_orient.device)
        z_angle[..., 0, 0] = 1.0
        z_angle[..., 1, 1] = 1.0
        z_angle[..., 2, 2] = 1.0
        z_angle.requires_grad_(True)
    
    params = [trans] + [z_angle] + [betas] + [pose_body]

    optimizer = torch.optim.LBFGS(
        params,
        max_iter=config["stages"]["chamfer"]["num_iters"],
        tolerance_grad=config["optimizer"]["tolerance_grad"],
        tolerance_change=config["optimizer"]["tolerance_change"],
        lr=0.1,
        line_search_fn="strong_wolfe",
    )
    root_orient.requires_grad_(False)

    iteration = 0
    def closure_stage_chamfer():
        optimizer.zero_grad()
        nonlocal iteration

        if config["stages"]["chamfer"]["yaw_lock"]:
            z_root_orient = compute_root_orient_z(z_angle) @ root_orient
        else:
            z_root_orient = rotation_6d_to_matrix(matrix_to_rotation_6d(z_angle))

        smpl_output = smpl_inference(
            poses=rotation_6d_to_matrix(matrix_to_rotation_6d(pose_body)),
            #betas=torch.repeat_interleave(torch.mean(betas, dim=0, keepdim=True), dim=0, repeats=betas.shape[0]),
            betas=torch.repeat_interleave(betas, dim=0, repeats=pose_body.shape[0]),
            root_orient=rotation_6d_to_matrix(matrix_to_rotation_6d(z_root_orient)),
            trans=trans,
        )
        
        root_orient_vel = so3_relative_angle(
            rotation_6d_to_matrix(matrix_to_rotation_6d(root_orient[1:, 0])),
            rotation_6d_to_matrix(matrix_to_rotation_6d(root_orient[:-1, 0])),
        )
        z_root_orient_vel = so3_relative_angle(
            rotation_6d_to_matrix(matrix_to_rotation_6d(z_root_orient[1:, 0])),
            rotation_6d_to_matrix(matrix_to_rotation_6d(z_root_orient[:-1, 0])),
        )

        """
        loss_chamfer = chamfer_distance_by_part(
            markers,
            smpl_output["vertices"],
            marker_labels=marker_labels,
            vertex_weights=smpl_inference.smpl.lbs_weights,
        )
        """

        loss = 0
        if "part_chamfer" in config["stages"]["chamfer"]["losses"]:
            loss_chamfer = chamfer_distance_by_part(
                markers,
                smpl_output["vertices"],
                marker_labels,
                smpl_inference.smpl.lbs_weights,
                single_directional=config["stages"]["chamfer"]["single_directional"],
            )
            loss += loss_chamfer * config["stages"]["chamfer"]["losses"]["part_chamfer"]
        if "full_chamfer" in config["stages"]["chamfer"]["losses"]:
            loss_chamfer = weighted_chamfer_distance(
                x=markers,
                y=smpl_output["vertices"],
                x_weights=get_marker_mask(markers),
                single_directional=config["stages"]["chamfer"]["single_directional"],
            )[0]
            loss += loss_chamfer * config["stages"]["chamfer"]["losses"]["full_chamfer"]
        if "root_orient_vel" in config["stages"]["chamfer"]["losses"]:
            loss += F.mse_loss(z_root_orient_vel, root_orient_vel) * config["stages"]["chamfer"]["losses"]["root_orient_vel"]
            import pdb; pdb.set_trace()
        if "reg_pose_body" in config["stages"]["chamfer"]["losses"]:
            reg_pos_body = F.mse_loss(pose_body, o_pose_body) #weighted_mse_loss(pose_body, o_pose_body, img_mask[:, None, None, None])
            loss += reg_pos_body * config["stages"]["chamfer"]["losses"]["reg_pose_body"]
        if "trans_vel" in config["stages"]["chamfer"]["losses"]:
            trans_vel = trans[1:] - trans[:-1]  # [F-1, 3]
            markers_mean = torch.mean(markers, dim=1)  # [F, 3]
            markers_vel = markers_mean[1:] - markers_mean[:-1]  # [F-1, 3]
            loss += F.mse_loss(trans_vel, markers_vel) * config["stages"]["chamfer"]["losses"]["trans_vel"]
        if "ground" in config["stages"]["chamfer"]["losses"]:
            ground_loss = torch.mean(F.relu(-smpl_output["joints"][..., 2]))
            #ground_loss = torch.mean(F.relu(-torch.min(smpl_output["joints"][..., 2])))
            loss += ground_loss * config["stages"]["chamfer"]["losses"]["ground"]
        if "reg_betas" in config["stages"]["chamfer"]["losses"]:
            loss += F.mse_loss(betas, o_betas) * config["stages"]["chamfer"]["losses"]["reg_betas"]

        loss.backward()
        
        if verbose:
            print("Chamfer", iteration, float(loss))

        if iter_fn is not None:
            iter_fn(
                stage="chamfer_" + str(repeat),
                iteration=iteration,
                initial_angle=np.array([initial_angle]),
                pose_body=rotation_6d_to_matrix(matrix_to_rotation_6d(pose_body)).detach().cpu().numpy(),
                betas=betas.detach().cpu().numpy(),
                trans=trans.detach().cpu().numpy(),
                root_orient=rotation_6d_to_matrix(matrix_to_rotation_6d(root_orient)).detach().cpu().numpy(),
            )

        iteration += 1
        return loss

    #root_orient[...] = compute_root_orient_z(z_angle).detach() @ root_orient
    #;root_orient.requires_grad_(True)
    optimizer.step(closure_stage_chamfer)
    if config["stages"]["chamfer"]["yaw_lock"]:
        root_orient[:] = compute_root_orient_z(z_angle) @ root_orient
    else:
        root_orient[:] = rotation_6d_to_matrix(matrix_to_rotation_6d(z_angle)) @ root_orient
    root_orient.detach()
    root_orient.requires_grad_(True)


def optim_markers(
    markers: torch.Tensor,
    pose_body: torch.Tensor,
    o_pose_body: torch.Tensor,
    betas: torch.Tensor,
    o_betas: torch.Tensor,
    root_orient: torch.Tensor,
    trans: torch.Tensor,
    barycentric_coords_one_hot: torch.Tensor,
    img_mask: torch.Tensor,  # [F]
    smpl_inference: SmplInference,
    config: Dict,
    initial_angle: float=0,
    repeat: int=0,
    verbose: bool=False,
    iter_fn: Callable=None,
):
    num_markers = markers.shape[1]

    if config["stages"]["marker"]["use_sdf"]:
        sdf = SDF(markers.device)

    virtual_markers = None
    if config["stages"]["marker"]["use_sdf"]:
        virtual_markers = sdf.barycentric_one_hot_to_points(barycentric_coords_one_hot)  # [M, 3]
        virtual_markers = torch.clone(virtual_markers).detach()
        virtual_markers.requires_grad_(True)
        params = [pose_body] + [betas] + [root_orient] + [trans] + [virtual_markers]
    else:
        params = [pose_body] + [betas] + [root_orient] + [trans]

    optimizer = torch.optim.LBFGS(
        params,
        max_iter=config["stages"]["marker"]["num_iters"],
        tolerance_grad=config["optimizer"]["tolerance_grad"],
        tolerance_change=config["optimizer"]["tolerance_change"],
        lr=1.0,
        line_search_fn="strong_wolfe",
    )

    iteration = 0
    def closure_stage_marker_pose():
        optimizer.zero_grad()
        nonlocal iteration
        nonlocal virtual_markers
        nonlocal barycentric_coords_one_hot

        smpl_output = smpl_inference(
            poses=rotation_6d_to_matrix(matrix_to_rotation_6d(pose_body)),
            betas=torch.repeat_interleave(betas, dim=0, repeats=pose_body.shape[0]),
            root_orient=rotation_6d_to_matrix(matrix_to_rotation_6d(root_orient)),
            trans=trans,
        )

        if config["stages"]["marker"]["use_sdf"]:
            barycentric_coords_one_hot = sdf.points_to_barycentric_one_hot(virtual_markers)

        vertices = torch.unsqueeze(smpl_output["vertices"], dim=1)  # [F, 1, V, 3]
        vertices = torch.repeat_interleave(vertices, repeats=num_markers, dim=1)  # [F, M, V, 3]
        bc_expanded = torch.unsqueeze(torch.unsqueeze(barycentric_coords_one_hot, dim=0), dim=-1)  # [1, M, V, 1]
        bc_expanded = torch.repeat_interleave(bc_expanded, repeats=vertices.shape[0], dim=0)  # [F, M, V, 1]
        bc_expanded = torch.repeat_interleave(bc_expanded, repeats=3, dim=-1)  # [F, M, V, 3]
        virtual_markers_expanded = vertices * bc_expanded  # [F, M, V, 3]
        virtual_markers_expanded = torch.sum(virtual_markers_expanded, dim=2)

        loss = 0
        if "marker" in config["stages"]["marker"]["losses"]:
            #marker_loss = (torch.norm(markers[f_index:f_index+window_size] - virtual_markers_expanded, dim=-1) - MARKER_DISTANCE) ** 2
            marker_loss = MarkerLoss(
                markers=markers,
                virtual_markers=virtual_markers_expanded,
                marker_weights=get_marker_mask(markers),
                marker_distance=MARKER_DISTANCE,
            )
            loss += torch.mean(marker_loss) * config["stages"]["marker"]["losses"]["marker"]
        if "reg_pose_body" in config["stages"]["marker"]["losses"]:
            reg_pos_body = F.mse_loss(pose_body, o_pose_body) #weighted_mse_loss(pose_body, o_pose_body, img_mask[:, None, None, None])
            loss += reg_pos_body * config["stages"]["marker"]["losses"]["reg_pose_body"]
        if "reg_betas" in config["stages"]["marker"]["losses"]:
            loss += F.mse_loss(betas, o_betas) * config["stages"]["marker"]["losses"]["reg_betas"]
        if "temporal" in config["stages"]["marker"]["losses"]:
            thetas_0 = pose_body[2:]
            thetas_1 = pose_body[1:-1]
            thetas_2 = pose_body[0:-2]
            vel = (thetas_0 - (2 * thetas_1) - thetas_2)
            temporal_loss = F.mse_loss(vel, torch.zeros_like(vel))
            loss += temporal_loss * config["stages"]["marker"]["losses"]["temporal"]
            import pdb; pdb.set_trace()

        loss.backward()

        if verbose:
            print("Marker", iteration, float(loss))

        if iter_fn is not None:
            iter_fn(
                stage="marker_" + str(repeat),
                iteration=iteration,
                initial_angle=np.array([initial_angle]),
                pose_body=rotation_6d_to_matrix(matrix_to_rotation_6d(pose_body)).detach().cpu().numpy(),
                betas=betas.detach().cpu().numpy(),
                trans=trans.detach().cpu().numpy(),
                root_orient=rotation_6d_to_matrix(matrix_to_rotation_6d(root_orient)).detach().cpu().numpy(),
            )

        iteration += 1
        return loss

    optimizer.step(closure_stage_marker_pose)

    pose_body=rotation_6d_to_matrix(matrix_to_rotation_6d(pose_body))
    root_orient=rotation_6d_to_matrix(matrix_to_rotation_6d(root_orient))


def compute_nearest_points(
    markers: torch.Tensor,
    pose_body: torch.Tensor,
    betas: torch.Tensor,
    root_orient: torch.Tensor,
    trans: torch.Tensor,
    smpl_inference: SmplInference,
    marker_labels: np.array,
    granularity: str,
    img_mask: torch.Tensor,
    device: torch.device,
    config: Dict,
    o_pose_body: torch.Tensor=None,
    window_size: int=1,
    use_velocity: bool=True,
):
    num_markers = markers.shape[1]
    num_joints = pose_body.shape[1]
    num_frames = markers.shape[0]

    smpl_output = smpl_inference(
        poses=rotation_6d_to_matrix(matrix_to_rotation_6d(pose_body)),
        betas=torch.repeat_interleave(torch.mean(betas, dim=0, keepdim=True), dim=0, repeats=betas.shape[0]),
        root_orient=rotation_6d_to_matrix(matrix_to_rotation_6d(root_orient)),
        trans=trans,
    )

    vertices = smpl_output["vertices"].detach().cpu().numpy()
    joints = smpl_output["joints"].detach().cpu().numpy()
    torch_faces = torch.from_numpy(smpl_inference.smpl.faces.astype(np.int64)).to(device)

    if o_pose_body is not None:
        o_smpl_output = smpl_inference(
            poses=rotation_6d_to_matrix(matrix_to_rotation_6d(pose_body)),
            betas=torch.repeat_interleave(torch.mean(betas, dim=0, keepdim=True), dim=0, repeats=betas.shape[0]),
            root_orient=rotation_6d_to_matrix(matrix_to_rotation_6d(root_orient)),
            trans=trans,
        )
        o_joints = o_smpl_output["joints"].detach().cpu().numpy()
        o_vertices = o_smpl_output["vertices"].detach().cpu().numpy()

    num_windows = int(np.ceil(num_frames / window_size))

    if granularity == "part":
        min_distance = np.array([np.inf] * num_joints)
    elif granularity == "marker":
        min_distance = np.array([np.inf] * num_markers)
    elif granularity == "full":
        min_distance = np.array([np.inf])
    min_distance = np.repeat(np.expand_dims(min_distance, axis=0), repeats=num_windows, axis=0)
    min_points = np.zeros((num_windows, num_markers, 3), dtype=np.float32)

    barycentric_coords_one_hot_final = torch.zeros((num_markers, 6890)).to(device)

    stride = window_size
    stride_num_frames = math.ceil(num_frames / stride)

    points_3d = np.zeros((stride_num_frames, num_markers, 3))
    distance = np.zeros((stride_num_frames, num_markers))
    vertex_indices = np.zeros((stride_num_frames, num_markers), dtype=np.int32)
    face_indices = np.zeros((stride_num_frames, num_markers), dtype=np.int32)

    if config["stages"]["compute_locations"]["use_mean"]:
        distance_matrix = np.zeros((stride_num_frames, num_markers, 6890), dtype=np.float32)

    valid_frames = torch.where(img_mask == 1)[0].tolist()
    
    window_index = 0
    for w_index in range(0, num_frames, window_size):
        w_start = w_index
        w_end = min(w_start + window_size, num_frames)
        for f_index in range(w_start, w_end, stride):
            fs_index = f_index // stride

            if fs_index not in valid_frames:
                continue

            if config["stages"]["compute_locations"]["use_mean"]:
                vertices_frame = vertices[f_index][None, :, :]
                markers_frame = markers[f_index].detach().cpu().numpy()[:, None, :]

                vertices_frame = np.repeat(vertices_frame, axis=0, repeats=markers.shape[1])
                markers_frame = np.repeat(markers_frame, axis=1, repeats=vertices.shape[1])

                distance_matrix[f_index] = np.linalg.norm(vertices_frame - markers_frame, axis=-1)

            mesh = trimesh.Trimesh(
                vertices=vertices[f_index],
                faces=smpl_inference.smpl.faces,
                process=False,
            )
            try:
                if config["stages"]["compute_locations"]["use_barycentric"]:
                    distance[fs_index], face_indices[fs_index], points_3d[fs_index] = igl.signed_distance(
                        markers[f_index].detach().cpu().numpy(),
                        vertices[f_index],
                        smpl_inference.smpl.faces.astype(np.int32),
                    )
                    distance[fs_index] = np.abs(distance[fs_index])
                elif config["stages"]["compute_locations"]["use_mean"]:
                    pass
                else:
                    closest_points = closest_point(
                        markers[f_index].detach().cpu().numpy(),
                        vertices[f_index],
                    )
                    vertex_indices[fs_index] = closest_points["vertex_indices"]
                    distance[fs_index] = closest_points["distances"]
                    points_3d[fs_index] = closest_points["points"]
            except Exception as e:
                print(e)
                import pdb; pdb.set_trace()

            fs_index = f_index // stride

        #try:
            barycentric_coords_one_hot = torch.zeros((markers.shape[1], vertices.shape[1])).to(device)  # [M, V]
            if config["stages"]["compute_locations"]["use_barycentric"]:
                with np.errstate(divide="raise"):
                    try:
                        barycentric_coords_np = trimesh.triangles.points_to_barycentric(mesh.triangles[face_indices[fs_index]], points_3d[fs_index])
                    except:
                        import pdb; pdb.set_trace()
                        import pdb; pdb.set_trace()

                barycentric_coords = torch.from_numpy(barycentric_coords_np).float().to(device)  # [M, 3]

                i0 = torch_faces[face_indices[fs_index]][:, 0]  # [M]
                i1 = torch_faces[face_indices[fs_index]][:, 1]  # [M]
                i2 = torch_faces[face_indices[fs_index]][:, 2]  # [M]

                barycentric_coords_one_hot.scatter_(1, i0.unsqueeze(1), barycentric_coords[:, [0]])  # [M, V]
                barycentric_coords_one_hot.scatter_(1, i1.unsqueeze(1), barycentric_coords[:, [1]])  # [M, V]
                barycentric_coords_one_hot.scatter_(1, i2.unsqueeze(1), barycentric_coords[:, [2]])  # [M, V]
            elif config["stages"]["compute_locations"]["use_mean"]:
                pass
            else:
                barycentric_coords_np = np.zeros((num_markers, 3))
                barycentric_coords_np[:, 0] = 1
                barycentric_coords = torch.from_numpy(barycentric_coords_np).float().to(device)

                i0 = torch.from_numpy(vertex_indices[fs_index].astype(np.int64)).to(device)
                i1 = i0.clone()
                i2 = i0.clone()

                barycentric_coords_one_hot.scatter_(1, i0.unsqueeze(1), barycentric_coords[:, [0]])  # [M, V]
                barycentric_coords_one_hot.scatter_(1, i1.unsqueeze(1), barycentric_coords[:, [1]])  # [M, V]
                barycentric_coords_one_hot.scatter_(1, i2.unsqueeze(1), barycentric_coords[:, [2]])  # [M, V]

            # points velocity
            vel_factor = np.ones((num_frames, num_markers))
            if use_velocity:
                i0_np = i0.detach().cpu().numpy()
                i1_np = i1.detach().cpu().numpy()
                i2_np = i2.detach().cpu().numpy()
                points_0 = o_vertices[:, i0_np]  # [F, M, 3]
                points_1 = o_vertices[:, i1_np]  # [F, M, 3]
                points_2 = o_vertices[:, i2_np]  # [F, M, 3]
                weights_0 = np.repeat(np.reshape(barycentric_coords_np[..., 0], (1, -1, 1)), repeats=num_frames, axis=0)
                weights_1 = np.repeat(np.reshape(barycentric_coords_np[..., 1], (1, -1, 1)), repeats=num_frames, axis=0)
                weights_2 = np.repeat(np.reshape(barycentric_coords_np[..., 2], (1, -1, 1)), repeats=num_frames, axis=0)
                points_markers = (points_0 * weights_0) + (points_1 * weights_1) + (points_2 * weights_2)
                points_markers_vel = points_markers[1:] - points_markers[:-1]
                points_markers_vel = np.concatenate((points_markers_vel[[0]] * 0, points_markers_vel), axis=0)
                markers_np = markers.detach().cpu().numpy()
                markers_vel = markers_np[1:] - markers_np[:-1]
                markers_vel = np.concatenate((markers_vel[[0]] * 0, markers_vel), axis=0)
                vel_factor = np.sum(markers_vel * points_markers_vel, axis=-1)  # [F, M]

            if granularity == "part":
                for m in range(num_joints):
                    part_distance = ((marker_labels[f_index] == m).astype(np.float32) * distance[fs_index])[marker_labels[f_index] == m]
                    temp_min_distance = np.median(part_distance)
                    if part_distance.size > 0 and temp_min_distance < min_distance[window_index, m]:
                        barycentric_coords_one_hot_final[marker_labels[f_index] == m] = barycentric_coords_one_hot[marker_labels[f_index] == m]
                        min_distance[window_index, m] = np.median(part_distance)
                        min_points[window_index, m] = points_3d[fs_index, m]
            elif granularity == "marker":
                for m in range(num_markers):
                    temp_min_distance = distance[fs_index, m]
                    if temp_min_distance < min_distance[window_index, m]:
                        barycentric_coords_one_hot_final[m] = barycentric_coords_one_hot[m]
                        min_distance[window_index, m] = distance[fs_index, m]
                        min_points[window_index, m] = points_3d[fs_index, m]
            elif granularity == "full":
                temp_min_distance = np.mean(distance[fs_index]) * np.mean(vel_factor[fs_index])
                if temp_min_distance < min_distance[window_index]:
                    barycentric_coords_one_hot_final = barycentric_coords_one_hot
                    min_distance[window_index] = np.mean(distance[fs_index])
                    min_points[window_index] = points_3d[fs_index]

        window_index += 1

    if config["stages"]["compute_locations"]["use_mean"]:
        # filter out frames without HMR SMPL output
        img_mask_np = img_mask.detach().cpu().numpy()
        distance_matrix_reduced = np.mean(distance_matrix[np.where(img_mask_np == 1)], axis=0)
        vertex_indices = np.argmin(distance_matrix_reduced, axis=-1)

        barycentric_coords_one_hot_final = torch.zeros((num_markers, 6890)).float().to(device)
        for marker_index in range(num_markers):
            barycentric_coords_one_hot_final[marker_index, vertex_indices[marker_index]] = 1.0  # [M, V]

    """
    median_points = np.zeros((num_markers, 3), dtype=np.float32)
    for m in range(num_markers):
        median_points[m] = geometric_median(min_points[:, m, :])
    
    if config["stages"]["compute_locations"]["use_barycentric"]:
        barycentric_coords_one_hot = trimesh.triangles.points_to_barycentric(mesh.triangles[face_indices[fs_index]], median_points)
        barycentric_coords_one_hot = torch.from_numpy(barycentric_coords_one_hot).float().to(device)
    
        i0 = torch_faces[face_indices[fs_index]][:, 0]  # [M]
        i1 = torch_faces[face_indices[fs_index]][:, 1]  # [M]
        i2 = torch_faces[face_indices[fs_index]][:, 2]  # [M]

        barycentric_coords_one_hot = torch.zeros((markers.shape[1], vertices.shape[1])).to(device)  # [M, V]
        barycentric_coords_one_hot.scatter_(1, i0.unsqueeze(1), barycentric_coords[:, [0]])  # [M, V]
        barycentric_coords_one_hot.scatter_(1, i1.unsqueeze(1), barycentric_coords[:, [1]])  # [M, V]
        barycentric_coords_one_hot.scatter_(1, i2.unsqueeze(1), barycentric_coords[:, [2]])  # [M, V]
    else:
        closest_points = closest_point(
            median_points.detach().cpu().numpy(),
            vertices[f_index],
        )
        vertex_indices[fs_index] = closest_points["vertex_indices"]

        barycentric_coords_np = np.zeros((num_markers, 3))
        barycentric_coords_np[:, 0] = 1
        barycentric_coords = torch.from_numpy(barycentric_coords_np).float().to(device)

        i0 = torch.from_numpy(vertex_indices[fs_index].astype(np.int64)).to(device)
        i1 = i0.clone()
        i2 = i0.clone()

        barycentric_coords_one_hot.scatter_(1, i0.unsqueeze(1), barycentric_coords[:, [0]])  # [M, V]
        barycentric_coords_one_hot.scatter_(1, i1.unsqueeze(1), barycentric_coords[:, [1]])  # [M, V]
        barycentric_coords_one_hot.scatter_(1, i2.unsqueeze(1), barycentric_coords[:, [2]])  # [M, V]
    """

    return barycentric_coords_one_hot_final


def compute_marker_labels_from_coords(
    smpl_inference: SmplInference,
    barycentric_coords_one_hot: torch.Tensor,  # [M, V]
    num_frames: int,
):
    lbs_weights = smpl_inference.get_lbs_weights()  # [V, J]
    vertex_ids = torch.argmax(lbs_weights, dim=-1)  # [V]
    coords_ids = torch.argmax(barycentric_coords_one_hot, dim=-1)
    marker_labels = vertex_ids[coords_ids]
    marker_labels = torch.repeat_interleave(
        marker_labels.unsqueeze(0),
        repeats=num_frames,
        dim=0,
    )
    return marker_labels


def compute_root_orient_y(
    angle: torch.Tensor,  # [..., J, 1]
) -> torch.Tensor:  # [..., J, 3, 3]
    y_vector = torch.zeros([*angle.shape[:-1]] + [3]).to(angle.device)
    y_vector[..., [1]] = angle
    y_vector = y_vector.to(angle.device)

    return axis_angle_to_matrix(y_vector)


def compute_root_orient_z(
    angle: torch.Tensor,  # [..., J, 1]
) -> torch.Tensor:  # [..., J, 3, 3]
    z_vector = torch.zeros([*angle.shape[:-1]] + [3]).to(angle.device)
    z_vector[..., [2]] = angle
    z_vector = z_vector.to(angle.device)

    return axis_angle_to_matrix(z_vector)


def chamfer_distance_by_part(
    markers: torch.Tensor,  # [F, M, 3]
    vertices: torch.Tensor,  # [F, V, 3]
    marker_labels: torch.Tensor,  # [F, M]
    vertex_weights: torch.Tensor,  # [V, P]
    single_directional: bool=False,
) -> torch.Tensor:
    vertex_mask = torch.argmax(vertex_weights, dim=-1)

    marker_labels_mode = torch.mode(marker_labels, dim=0)[0]

    loss = 0
    for i in torch.unique(marker_labels_mode).tolist():
        vertices_part = vertices[:, vertex_mask == i]
        markers_part = markers[:, marker_labels_mode == i]
        loss_part = (chamfer_distance(vertices_part, markers_part, single_directional=single_directional)[0] - MARKER_DISTANCE) ** 2
        loss += loss_part

    return loss


def get_marker_mask(
    markers: torch.Tensor, # [F, M, 3]
) -> torch.Tensor:  # [F, M]
    """
    Returns a mask for each marker if at origin

    Args:
        markers: marker positions [F, M, 3]

    Returns:
        torch.Tensor: [F, M]
    """
    return (torch.sum(torch.abs(markers), axis=-1) != 0.0)


def weighted_mse_loss(
    input: torch.Tensor, # [N, ...]
    target: torch.Tensor,  # [N, ...]
    weights: torch.Tensor, # [N, ...]
):
    loss = F.mse_loss(input, target, reduction="none")  # [N, ...]
    return torch.mean(loss * weights)
