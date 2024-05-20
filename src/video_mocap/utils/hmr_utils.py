from collections.abc import Callable
from typing import Optional, Dict

import numpy as np
from pytorch3d.loss.chamfer import chamfer_distance
import torch
import torch.nn.functional as F

from video_mocap.optimization import compute_root_orient_y
from video_mocap.utils.smpl import SmplInference


# From https://github.com/shubham-goel/4D-Humans/blob/main/hmr2/utils/geometry.py
def perspective_projection(points: torch.Tensor,
                           translation: torch.Tensor,
                           focal_length: torch.Tensor,
                           camera_center: Optional[torch.Tensor] = None,
                           rotation: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the perspective projection of a set of 3D points.
    Args:
        points (torch.Tensor): Tensor of shape (B, N, 3) containing the input 3D points.
        translation (torch.Tensor): Tensor of shape (B, 3) containing the 3D camera translation.
        focal_length (torch.Tensor): Tensor of shape (B, 2) containing the focal length in pixels.
        camera_center (torch.Tensor): Tensor of shape (B, 2) containing the camera center in pixels.
        rotation (torch.Tensor): Tensor of shape (B, 3, 3) containing the camera rotation.
    Returns:
        torch.Tensor: Tensor of shape (B, N, 2) containing the projection of the input points.
    """
    batch_size = points.shape[0]
    if rotation is None:
        rotation = torch.eye(3, device=points.device, dtype=points.dtype).unsqueeze(0).expand(batch_size, -1, -1)
    if camera_center is None:
        camera_center = torch.zeros(batch_size, 2, device=points.device, dtype=points.dtype)
    # Populate intrinsic camera matrix K.
    K = torch.zeros([batch_size, 3, 3], device=points.device, dtype=points.dtype)
    K[:,0,0] = focal_length[:,0]
    K[:,1,1] = focal_length[:,1]
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]


# From https://github.com/brjathu/PHALP/blob/master/phalp/models/hmar/hmar.py
# From https://github.com/brjathu/PHALP/blob/master/phalp/trackers/PHALP.py
def get_3d_parameters(
    smpl_inference,
    pred_smpl_betas,  # [N, 10]
    pred_smpl_body_pose,  # [N, 23, 3, 3]
    pred_smpl_global_orient,  # [N, 1, 3, 3]
    pred_cam,  # [N, 3]
    center,  # [N, 2]
    size,  # [N, 2]
    scale,  # [N, 1]
):
    FOCAL_LENGTH = 5000  # defaults used in HMR 2.0

    device = pred_cam.device
    dtype = pred_cam.dtype

    img_size = 256

    img_height = size[:, [0]]  # [N, 1]
    img_width = size[:, [1]]  # [N, 1]
    new_image_size = torch.max(size, dim=-1, keepdim=True)[0]  # [N, 1]
    top, left = (new_image_size - img_height) // 2, (new_image_size - img_width) // 2
    ratio = 1.0 / torch.round(new_image_size) * img_size  # [N, 1]
    center = (center + torch.cat((left, top), dim=-1).to(device)) * ratio  # [N, 2]

    scale = scale * new_image_size * ratio  # [N, 1]

    smpl_betas = pred_smpl_betas  # [N, 10]
    smpl_body_pose = pred_smpl_body_pose  # [N, 23, 3, 3]
    smpl_global_orient = pred_smpl_global_orient  # [N, 1, 3, 3]
    smpl_translation = torch.zeros((smpl_betas.shape[0], 3)).to(device).to(smpl_betas.device)  # [N, 1, 3, 3]
    pred_cam = pred_cam  # [N, 3]

    batch_size = pred_cam.shape[0]
    focal_length = FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)  # [N, 2]

    smpl_output = smpl_inference(
        smpl_body_pose,
        smpl_betas,
        smpl_global_orient,
        smpl_translation,
    )
    pred_joints = smpl_output["joints"]

    pred_cam_temp = torch.stack([pred_cam[:,1], pred_cam[:,2], 2*focal_length[:, 0]/(pred_cam[:,0]*scale[:, 0] + 1e-9)], dim=1)
    pred_cam_t = torch.cat((
        pred_cam_temp[:, :2] + (center-img_size/2.0) * pred_cam_temp[:, [2]] / focal_length,
        pred_cam_temp[:, [2]],
    ), dim=1)

    camera_center = torch.zeros(batch_size, 2)
    pred_keypoints_2d_smpl = perspective_projection(
        points=pred_joints,
        translation=pred_cam_t.to(device),
        focal_length=focal_length / img_size,
        camera_center=camera_center.to(device),
        rotation=torch.eye(3,).unsqueeze(0).expand(batch_size, -1, -1).to(device),
    )

    pred_keypoints_2d_smpl = (pred_keypoints_2d_smpl+0.5)*img_size

    output = {}
    output["camera_center"] = camera_center.to(device)
    output["focal_length"] = focal_length / img_size
    output["pred_cam_t"] = pred_cam_t
    output["pred_joints"] = pred_joints
    output["pred_keypoints_2d_smpl"] = pred_keypoints_2d_smpl / img_size
    output["rotation"] = torch.eye(3,).unsqueeze(0).expand(batch_size, -1, -1).to(device)
    return output


def convert_hmr_pos_to_mocap_pos(pos):
    x = pos[..., [0]]  # [A, F, 1]
    y = pos[..., [2]]  # [A, F, 1]
    z = pos[..., [1]] * -1  # [A, F, 1]
    #pos[..., [0, 1, 2]] = pos[..., [0, 2, 1]]
    #pos[..., 2] *= -1
    return torch.cat((x, y, z), dim=-1)


def convert_mocap_pos_to_hmr_pos(pos):
    x = pos[..., [0]]  # [A, F, 1]
    y = pos[..., [2]] * -1  # [A, F, 1]
    z = pos[..., [1]]  # [A, F, 1]
    #pos[..., 2] *= -1
    #pos[..., [0, 2, 1]] = pos[..., [0, 1, 2]]
    return torch.cat((x, y, z), dim=-1)


def matrix_33_to_44(mat):
    new_shape = list(mat.shape)
    new_shape[-1] = 4
    new_shape[-2] = 4

    output = torch.zeros(new_shape, dtype=mat.dtype, device=mat.device)
    output[..., :3, :3] = mat
    output[..., 3, 3] = 1
    return output


def matrix_44_to_33(mat):
    return mat[... :3, :3]


def apply_matrix_33_to_vector_3(
    mat,  # [..., 3, 3]
    vec,  # [..., 3]
):
    mat_expanded = matrix_33_to_44(mat)  # [..., 4, 4]
    vec_expanded = torch.nn.functional.pad(vec, (0, 1), mode="constant", value=1.0)  # [..., 4, 1]
    output = mat_expanded @ vec_expanded[..., None]
    return output[..., :3, 0]


def optim_reprojection(
    markers: torch.Tensor,  # [F, M, 3]
    pose_body: torch.Tensor,  # [F, J-1, 3, 3]
    betas: torch.Tensor,  # [1, 10]
    hmr_betas: torch.Tensor,  # [F, 10]
    root_orient: torch.Tensor,  # [F, 1, 3, 3]
    trans: torch.Tensor,  # [F, 3]
    pred_cam: torch.Tensor,  # [F, 3]
    cam_center: torch.Tensor,  # [F, 2]
    cam_size: torch.Tensor,  # [F, 2]
    cam_scale: torch.Tensor,  # [F, 1]
    angle: float,
    img_mask: torch.Tensor,  # [F]
    smpl_inference: SmplInference,
    num_iters: int,
    config: Dict,
    verbose: bool=False,
    iter_fn: Callable=None,
) -> Dict:
    """
    Args:
        markers: marker positions [F, M, 3]
        pose_body: HMR 2.0 poses for SMPL mesh without root [F, J-1, 3, 3]
        betas: mean HMR 2.0 betas [1, 10]
        hmr_betas: HMR 2.0 betas [F, 10]
        root_orient: HMR 2.0 root orientation (different axes from markers) [F, 1, 3, 3]
        trans: HMR 2.0 translation [F, 3]
        pred_cam: HMR 2.0 camera_bbox [F, 3]
        cam_center: HMR 2.0 camera center [F, 2]
        cam_size: HMR 2.0 camera size [F, 2]
        cam_scale: HMR 2.0 camera scale [F, 2]
        angle: yaw rotation offset
        smpl_inference: SMPL inference object
        config: config dictionary
        verbose: prints loss is True

    Returns:
        Dict:
            root_orient: root orientation with axes consistent with HMR 2.0 [F, 3]
    """
    device = markers.device
    num_frames = pose_body.shape[0]

    # copy parameters
    pose_body = torch.clone(pose_body)
    betas = torch.clone(betas).detach()
    root_orient = torch.clone(root_orient)
    trans = torch.clone(trans)

    pred_cam = torch.clone(pred_cam)
    cam_center = torch.clone(cam_center)
    cam_size = torch.clone(cam_size)
    cam_scale = torch.clone(cam_scale)

    correction_matrix = torch.tensor([[[
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
    ]]]).to(device).float()  # [1, 1, 3, 3]
    correction_matrix = torch.repeat_interleave(correction_matrix, repeats=num_frames, dim=0)

    joints_output = get_3d_parameters(
        smpl_inference,
        pred_smpl_betas=hmr_betas,
        pred_smpl_body_pose=pose_body,
        pred_smpl_global_orient=root_orient,
        pred_cam=pred_cam,
        center=cam_center,
        size=cam_size,
        scale=cam_scale,
    )

    camera_center = joints_output["camera_center"]

    reproject_2d_joints = joints_output["pred_keypoints_2d_smpl"]  # [F, J, 2]
    reproject_2d_joints = torch.repeat_interleave(reproject_2d_joints[None], repeats=1, dim=0)  # [A, F, J, 2]
    reproject_2d_joints = torch.nan_to_num(reproject_2d_joints, 0)

    cam_translation = joints_output["pred_cam_t"]
    reproject_mask = (cam_translation == cam_translation)  # [F, 3]
    reproject_mask = torch.mean(reproject_mask.float(), dim=-1).detach()  # [F]
    cam_translation = torch.nan_to_num(cam_translation, 0)

    # swap SMPL and camera positions
    temp_translations = torch.clone(trans).detach()
    pred_smpl_translations = cam_translation
    cam_translation = temp_translations

    # optimize camera placement
    translation_offset = convert_mocap_pos_to_hmr_pos(torch.median(torch.reshape(markers, [-1, 3]), dim=0, keepdim=True)[0]) -\
        torch.median(pred_smpl_translations, dim=0, keepdim=True)[0]
    pred_smpl_translations = pred_smpl_translations + translation_offset  # reposition SMPL mesh to markers
    pred_smpl_translations = torch.repeat_interleave(pred_smpl_translations[None], repeats=1, dim=0)  # [A, F, 3]
    pred_smpl_translations.requires_grad_(True)

    # moves camera such that markers are at (0, 0, 0)
    cam_translation_single = torch.mean(cam_translation - translation_offset, dim=0, keepdim=True)  # [1, 3]
    cam_translation_single = torch.repeat_interleave(cam_translation_single, repeats=1, dim=0)  # [A, 3]
    cam_translation_single.requires_grad_(True)

    y_angle_single = torch.ones(1, 1, 1, 1)*angle  # [A, 1, 1, 1]
    y_angle_single = y_angle_single.to(device).requires_grad_(True)
    focal_length = torch.mean(joints_output["focal_length"], dim=0, keepdim=True)  # [1, 2]

    y_angle = torch.repeat_interleave(y_angle_single, repeats=num_frames, dim=1)  # [A, F, 1, 1]
    pred_keypoints_2d_smpl = torch.clone(reproject_2d_joints)  # [F, 90]
    y_root_orient = compute_root_orient_y(y_angle) @ root_orient
    camera_offset = pred_smpl_translations - cam_translation  # center SMPL around camera
    inv_translation = apply_matrix_33_to_vector_3(compute_root_orient_y(-y_angle)[:, 0], camera_offset)  # [A, F, 3], rotate SMPL around camera
    inv_translation = inv_translation + torch.repeat_interleave(cam_translation[None, ...], dim=0, repeats=1)  # [A, F, 3], move SMPL back to world coordinates

    optimizer = torch.optim.LBFGS(
        [y_angle_single] + [pred_smpl_translations] + [cam_translation_single] + [betas],
        max_iter=num_iters,
        tolerance_grad=config["optimizer"]["tolerance_grad"],
        tolerance_change=config["optimizer"]["tolerance_change"],
        lr=1.0,
        line_search_fn="strong_wolfe",
    )

    pose_body_expanded = torch.repeat_interleave(pose_body[None, ...], repeats=1, dim=0)  # [A, F, J-1, 3, 3]
    betas = torch.repeat_interleave(betas, repeats=num_frames, dim=0)
    betas_expanded = torch.repeat_interleave(betas[None, ...], repeats=1, dim=0)  # [A, F, 10]

    iteration = 0
    def closure():
        optimizer.zero_grad()
        nonlocal iteration
        nonlocal pred_smpl_translations
        nonlocal pred_keypoints_2d_smpl
        nonlocal cam_translation
        nonlocal y_root_orient
        nonlocal inv_translation

        cam_translation = torch.repeat_interleave(cam_translation_single[:, None], dim=1, repeats=num_frames)  # [A, F, 3]
        y_angle = torch.repeat_interleave(y_angle_single, repeats=num_frames, dim=1)  # [A, F, 1, 1]
        y_root_orient = compute_root_orient_y(y_angle) @ root_orient  # [A, F, 1, 3, 3]

        camera_offset = pred_smpl_translations - cam_translation  # [A, F, 3]
        inv_translation = apply_matrix_33_to_vector_3(compute_root_orient_y(-y_angle)[:, 0], camera_offset)  # [A, F, 3]
        inv_translation = inv_translation + cam_translation  # [A, F, 3]

        smpl_output = smpl_inference(
            torch.flatten(pose_body_expanded, start_dim=0, end_dim=1),
            torch.flatten(betas_expanded, start_dim=0, end_dim=1),
            root_orient,
            torch.flatten(inv_translation, start_dim=0, end_dim=1),
        )

        pred_keypoints_2d_smpl = perspective_projection(
            points=smpl_output["joints"],  # [A*F, 45, 3]
            translation=torch.flatten(cam_translation, start_dim=0, end_dim=1),  # [A*F, 3]
            focal_length=torch.repeat_interleave(focal_length, dim=0, repeats=1*num_frames),  # [A*F, 3]
            camera_center=torch.repeat_interleave(joints_output["camera_center"], dim=0, repeats=1),  # [A*F, 2]
            rotation=torch.eye(3,).unsqueeze(0).expand(1*num_frames, -1, -1).to(device),  # [A*F, 3, 3]
        ).reshape((1, num_frames, 45, 2)) + 0.5  # [A, F, 45, 2]
        reproject_loss = torch.mean((pred_keypoints_2d_smpl - reproject_2d_joints)**2 * reproject_mask[None, :, None, None]) * config["stages"]["reprojection_part"]["losses"]["reprojection"]

        x = pred_smpl_translations[..., [0]]  # [A, F, 1]
        y = pred_smpl_translations[..., [2]]  # [A, F, 1]
        z = pred_smpl_translations[..., [1]] * -1  # [A, F, 1]
        corrected_pred_smpl_translations = torch.cat((x, y, z), dim=-1)  # [A, F, 3]
        smpl_output = smpl_inference(
            torch.flatten(pose_body_expanded, start_dim=0, end_dim=1),  # [A*F, J-1, 3, 3]
            torch.flatten(betas_expanded, start_dim=0, end_dim=1),  # [A*F, 1, 3, 3]
            torch.flatten(correction_matrix @ y_root_orient, start_dim=0, end_dim=1),  # [A*F, 1, 3, 3]
            torch.flatten(corrected_pred_smpl_translations, start_dim=0, end_dim=1),  # [A*F, 3]
        )

        pred_vertices = smpl_output["vertices"]  # [A*F, V, 3]
        chamfer_loss = chamfer_distance(
            torch.repeat_interleave(markers, repeats=1, dim=0),
            pred_vertices,
            single_directional=True,
        )[0] * config["stages"]["reprojection_part"]["losses"]["chamfer"]
        loss = reproject_loss + chamfer_loss
        loss.backward()
        if verbose:
            print("Reprojection", iteration, float(loss), float(reproject_loss), np.rad2deg(angle.item()), np.rad2deg(y_angle_single.item()))

        if iter_fn is not None:
            iter_fn(
                stage="reprojection",
                iteration=iteration,
                pose_body=pose_body.detach().cpu().numpy(),
                betas=betas.detach().cpu().numpy(),
                trans=corrected_pred_smpl_translations[0].detach().cpu().numpy(),
                root_orient=(correction_matrix @ y_root_orient)[0].detach().cpu().numpy(),
                pred_angle=y_angle_single.item(),
                initial_angle=angle,
                pred_2d_joints=pred_keypoints_2d_smpl[0].detach().cpu().numpy(),
                gt_2d_joints=reproject_2d_joints[0].detach().cpu().numpy(),
            )

        iteration += 1
        return loss

    optimizer.step(closure)

    pose_body_expanded = torch.repeat_interleave(pose_body[None, ...], repeats=1, dim=0)  # [A, F, J-1, 3, 3]
    betas_expanded = torch.repeat_interleave(betas[None, ...], repeats=1, dim=0)  # [A, F, 10]

    smpl_output = smpl_inference(
        torch.flatten(pose_body_expanded, start_dim=0, end_dim=1),  # [A*F, J-1, 3, 3]
        torch.flatten(betas_expanded, start_dim=0, end_dim=1),  # [A*F, 1, 3, 3]
        root_orient,  # [F, 1, 3, 3]
        torch.flatten(inv_translation, start_dim=0, end_dim=1),  # [A*F, 3]
    )

    pred_keypoints_2d_smpl = perspective_projection(
        points=smpl_output["joints"],  # [A*F, 3]
        translation=torch.flatten(cam_translation, start_dim=0, end_dim=1),  # [A*F, 3]
        focal_length=torch.repeat_interleave(focal_length, dim=0, repeats=1*num_frames),  # [A*F, 3]
        camera_center=torch.repeat_interleave(joints_output["camera_center"], dim=0, repeats=1),  # [A*F, 2]
        rotation=torch.eye(3,).unsqueeze(0).expand(1*num_frames, -1, -1).to(device),  # [A*F, 3, 3]
    ).reshape((1, num_frames, 45, 2)) + 0.5  # [A, F, 45, 2]

    smpl_output_world = smpl_inference(
        torch.flatten(pose_body_expanded, start_dim=0, end_dim=1),  # [A*F, J-1, 3, 3]
        torch.flatten(betas_expanded, start_dim=0, end_dim=1),  # [A*F, 1, 3, 3]
        torch.flatten(correction_matrix @ y_root_orient, start_dim=0, end_dim=1),  # [F, 1, 3, 3]
        torch.flatten(inv_translation, start_dim=0, end_dim=1),  # [A*F, 3]
    )

    reproject_errors = torch.mean(
        ((pred_keypoints_2d_smpl[0] - reproject_2d_joints[0]) ** 2) * reproject_mask[None, :, None, None]
    ).item()
    chamfer_errors = chamfer_distance(
        markers,
        torch.reshape(smpl_output_world["vertices"], (1, num_frames, -1, 3))[0],
        single_directional=True,
    )[0].item()

    pred_smpl_translations = pred_smpl_translations.requires_grad_(False)
    trans = convert_hmr_pos_to_mocap_pos(pred_smpl_translations)
    cam_translation = convert_hmr_pos_to_mocap_pos(cam_translation)
    root_orient = correction_matrix @ y_root_orient  # [A, F, 1, 3, 3]

    output = {}
    output["pose_body"] = torch.clone(pose_body_expanded).detach()  # [A, F, J-1, 3, 3]
    output["betas"] = torch.clone(betas_expanded).detach()  # [A, F, 10]
    output["root_orient"] = torch.clone(root_orient).detach()  # [A, F, 1, 3, 3]
    output["trans"] = torch.clone(trans).detach()  # [A, F, 3]
    output["joints_2d"] = torch.clone(pred_keypoints_2d_smpl).detach()
    output["joints_2d_gt"] = reproject_2d_joints
    output["cam_trans"] = torch.clone(cam_translation)  # [1, F, 3]
    output["camera_center"] = torch.clone(camera_center)  # [F, 2]
    output["focal_length"] = torch.clone(focal_length)  # [1, 2]
    output["reproject_mask"] = torch.clone(reproject_mask)
    output["input_angle"] = angle.item()
    output["output_angle"] = y_angle_single.item()
    output["metrics"] = {
        "chamfer": chamfer_errors,
        "reproject": reproject_errors,
    }
    return output
