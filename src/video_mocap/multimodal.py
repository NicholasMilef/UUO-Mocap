import os
from typing import Dict

import numpy as np
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d, matrix_to_quaternion, quaternion_to_matrix
import roma
import torch

from video_mocap.img_smpl.img_smpl import ImgSmpl
from video_mocap.markers.markers import Markers
from video_mocap.markers.markers_utils import filter_rigid, find_best_part_fits, segment_markers, segment_rigid
from video_mocap.models.marker_segmenter_multimodal import MarkerSegmenterMultimodal
from video_mocap.optimization import compute_marker_labels_from_coords, compute_nearest_points, optim_markers, optim_root, optim_chamfer, compute_root_orient_z, weighted_chamfer_distance, get_marker_mask
from video_mocap.utils.aabb import get_aabb, get_aabb_volume
from video_mocap.utils.hmr_utils import optim_reprojection
from video_mocap.utils.smpl import SmplInference
from video_mocap.utils.smpl_utils import get_joint_name
from video_mocap.vis.visualize_part import visualize_part


def visualize_parts(subtree, markers, vertices, faces, marker_labels, marker_indices, vertex_indices):
    format = ".gif"
    visualize_part(
        os.path.join("render_output", "___".join([str(get_joint_name(x)) for x in subtree]) + format),
        markers.detach().cpu().numpy(),
        vertices.detach().cpu().numpy(),
        faces.detach().cpu().numpy(),
        marker_labels.detach().cpu().numpy(),
        marker_indices.detach().cpu().numpy(),
        vertex_indices.detach().cpu().numpy(),
    )


def normalize_rot(rot):
    return rotation_6d_to_matrix(matrix_to_rotation_6d(rot))


def multimodal_video_mocap(
    img_smpl: ImgSmpl,
    mocap_markers: Markers,
    device: torch.device,
    config: Dict,
    offset: int=None,
    print_options=[],
    save_stages: bool=False,
    save_iterations: bool=False,
    visualize_fits: bool=False
) -> Dict:
    """
    Core framework for video_mocap

    Args:
        img_smpl: SMPL produced by monocular HMR
        mocap_markers: motion capture markers
        device: PyTorch device
        config: configuration file
        offset: offset (temporal offset if unsynchronized video and mocap)
        print_options: prints debug information ["loss"]
        save_stages: saves results of stages
        save_iteractions:
            saves each optimization iteration result. This is
            very memory intensive, so it should be used only in visualization
            applications.
        visualize_fits: visualize output of each part fit
            
    Returns:
        Dict: contains output SMPL parameters and other important information
            markers_labels: mocap labels after segmentation labeling
            mocap_frame_rate: mocap framerate
            mocap_markers: mocap marker positions

            betas: optimized SMPL beta values
            pose_body: optimized SMPL pose_body rotations
            root_orient: optimized SMPL root orientation
            trans: optimized SMPL translation

            stages: intermediate results for each stage
                find_best_part_fits: part optimization
                root: root optimization
                chamfer: chamfer (pose fitting) optimization
                marker: marker (inverse kinematics) optimization

            [deprecated] chain: output chain of joints from finding parts
    """
    smpl_inference = SmplInference(device)


    o_trans = img_smpl.trans.clone().detach().to(device)
    o_root_orient = img_smpl.root_orient.clone().detach().to(device)
    o_pose_body = img_smpl.pose_body.clone().detach().to(device)
    o_betas = torch.sum(img_smpl.betas, dim=0, keepdim=True).clone().detach().to(device)
    o_betas = o_betas / torch.sum(img_smpl.img_mask)
    o_foot_contacts = img_smpl.foot_contacts.clone().detach().to(device)

    o_hmr_root_orient = img_smpl.hmr_root_orient.clone().detach().to(device)
    o_hmr_betas = img_smpl.betas.clone().detach().to(device)
    o_camera_bbox = img_smpl.camera_bbox.clone().detach().to(device)
    o_center = img_smpl.center.clone().detach().to(device)
    o_scale = img_smpl.scale.clone().detach().to(device)
    o_size = img_smpl.size.clone().detach().to(device)

    if save_iterations:
        global iter_output
        iter_output = {}
        iter_output["input"] = {"markers": mocap_markers.get_points()}
        def save_iter_fn(stage, iteration, **kwargs):
            if stage not in iter_output:
                iter_output[stage] = {}

            if "initial_angle" in kwargs:
                if kwargs["initial_angle"].item() not in iter_output[stage]:
                    iter_output[stage][kwargs["initial_angle"].item()] = {}
                iter_output[stage][kwargs["initial_angle"].item()][iteration] = {}
            elif "part" in kwargs:
                if kwargs["part"] not in iter_output[stage]:
                    iter_output[stage][kwargs["part"]] = {}
                iter_output[stage][kwargs["part"]][iteration] = {}
            else:
                iter_output[stage][iteration] = {}

            pkl_parameters = [
                "betas",
                "pose_body",
                "trans",
                "root_orient",
                "markers",
                "pred_angle",
                "pred_2d_joints",
                "pred_2d_joints",
                "gt_2d_joints",
                "part_joints",
            ]
            for smpl_parameter in pkl_parameters:
                if smpl_parameter in kwargs:
                    if "initial_angle" in kwargs:
                        iter_output[stage][kwargs["initial_angle"].item()][iteration][smpl_parameter] = kwargs[smpl_parameter]
                    elif "part" in kwargs:
                        iter_output[stage][kwargs["part"]][iteration][smpl_parameter] = kwargs[smpl_parameter]
                    else:
                        iter_output[stage][iteration][smpl_parameter] = kwargs[smpl_parameter]                
    else:
        save_iter_fn = None

    # pad out translation
    if mocap_markers.get_frequency() != img_smpl.freq:
        o_trans_list = []
        o_root_orient_list = []
        o_pose_body_list = []
        o_foot_contacts_list = []

        new_num_frames = round(o_trans.shape[0] * (mocap_markers.get_frequency() / img_smpl.freq))
        for i in range(new_num_frames):
            frame = int(i * (img_smpl.freq / mocap_markers.get_frequency()))
            alpha = i * (img_smpl.freq / mocap_markers.get_frequency()) - frame
            inv_alpha = 1.0 - alpha

            if frame + 1 < o_trans.shape[0]:
                o_trans_list.append((o_trans[frame + 1] * alpha) + (o_trans[frame] * inv_alpha))
                o_foot_contacts_list.append((o_foot_contacts[frame + 1] * alpha) + (o_foot_contacts[frame] * inv_alpha))

                root_orient_0 = matrix_to_quaternion(o_root_orient[frame])
                root_orient_1 = matrix_to_quaternion(o_root_orient[frame + 1])

                pose_body_0 = matrix_to_quaternion(o_pose_body[frame])
                pose_body_1 = matrix_to_quaternion(o_pose_body[frame + 1])

                alpha = torch.tensor([alpha]).to(device)
                root_orient_interpolate = roma.utils.unitquat_slerp(root_orient_0, root_orient_1, alpha)
                pose_body_interpolate = roma.utils.unitquat_slerp(pose_body_0, pose_body_1, alpha)

                o_root_orient_list.append(quaternion_to_matrix(root_orient_interpolate)[0])
                o_pose_body_list.append(quaternion_to_matrix(pose_body_interpolate)[0])
            else:
                o_trans_list.append(o_trans[frame])
                o_root_orient_list.append(o_root_orient[frame])
                o_pose_body_list.append(o_pose_body[frame])
                o_foot_contacts_list.append(o_foot_contacts[frame])

        o_trans = torch.stack(o_trans_list, dim=0)
        o_root_orient = torch.stack(o_root_orient_list, dim=0)
        o_pose_body = torch.stack(o_pose_body_list, dim=0)
        o_foot_contacts = torch.stack(o_foot_contacts_list, dim=0)

    trans = o_trans.clone().detach().to(device).requires_grad_(True)
    root_orient = o_root_orient.clone().detach().to(device).requires_grad_(True)

    markers = torch.from_numpy(mocap_markers.get_points()).float().to(device)
    markers = torch.nan_to_num(markers, nan=0)

    # temporary: allows for loss function
    min_frames = min(markers.shape[0], trans.shape[0])
    markers = markers[:min_frames]
    o_trans = o_trans[:min_frames]
    o_root_orient = o_root_orient[:min_frames]
    o_pose_body = o_pose_body[:min_frames]
    o_betas = o_betas
    o_foot_contacts = o_foot_contacts[:min_frames]
    trans = trans[:min_frames]
    root_orient = root_orient[:min_frames]

    ########## Stage: compute temporal alignment ##########
    if "progress" in print_options:
        print("Stage: computing temporal alignment...")
    # subsample to 30 Hz

    if offset is None:
        offset = 0

    o_pose_body = pad(o_pose_body, offset).detach()
    o_betas = o_betas.detach()
    o_root_orient = pad(o_root_orient, offset).detach()
    o_trans = pad(o_trans, offset).detach()
    o_foot_contacts = pad(o_foot_contacts, offset).detach()
    markers = pad(markers, -offset).detach()
    num_frames = trans.shape[0]

    ########## Stage: marker segmentation ##########
    print("Stage: computing marker segmentation...")
    if True:
        with torch.no_grad():
            if config["stages"]["part"]["mode"] == "cluster":
                segmented_groups = segment_rigid(markers.detach().cpu().numpy())
                segmented_markers = torch.zeros((markers.shape[:2]))

                group_index = 0
                for group in segmented_groups:
                    segmented_markers[:, group] = group_index
                    group_index += 1

                segmented_markers = segmented_markers.long().to(device)  # [F, M]

        mean_img_smpl_output = smpl_inference(
            poses=o_pose_body,
            betas=o_betas * 0,
            root_orient=o_root_orient,
            trans=o_trans * 0,  # zero out translation
        )
        aabb_volume_ratio = torch.median(get_aabb_volume(get_aabb(markers)) / \
            get_aabb_volume(get_aabb(mean_img_smpl_output["vertices"])))

        filter_output = None
        joints_2d_gt = None
        focal_length = None
        reproject_mask = None
        cam_trans = None
        camera_center = None

        if config["find_best_part_fits"]:
            trans = torch.median(markers, dim=1)[0].requires_grad_(True)
            root_orient = o_root_orient.clone().requires_grad_(True)
            betas = o_betas.clone().requires_grad_(True)

            if config["stages"]["reprojection_part"]["num_iters"] > 0:
                num_angles = config["stages"]["reprojection_part"]["num_angles"]  # A
                angles = torch.arange(0, 2*np.pi, (2*np.pi)/num_angles)

                reproject_output = {}
                reproject_output["pose_body"] = []
                reproject_output["betas"] = []
                reproject_output["root_orient"] = []
                reproject_output["trans"] = []
                reproject_output["joints_2d"] = []
                reproject_output["joints_2d_gt"] = []
                reproject_output["focal_length"] = []
                reproject_output["reproject_mask"] = []
                reproject_output["camera_center"] = []
                reproject_output["cam_trans"] = []
                reproject_output["input_angle"] = []
                reproject_output["output_angle"] = []
                reproject_output["metrics"] = []

                for angle_index in range(num_angles):
                    angle = angles[angle_index]
                    reproject_output_single = optim_reprojection(
                        markers=markers,  # [F, M, 3], requires_grad=False
                        pose_body=o_pose_body,  # [F, J-1, 3, 3], requires_grad=False
                        betas=betas,  # [1, 10], requires_grad=True
                        hmr_betas=o_hmr_betas,  # [F, 10], requires_grad=False
                        root_orient=o_hmr_root_orient,  # [F, 1, 3, 3], requires_grad=True
                        trans=trans,  # [F, 3], requires_grad=True
                        pred_cam=o_camera_bbox,  # [F, 3], requires_grad=False
                        cam_center=o_center,  # [F, 3], requires_grad=False
                        cam_scale=o_scale,  # [F, 3], requires_grad=False
                        cam_size=o_size,  # [F, 3], requires_grad=False
                        angle=angle,
                        img_mask=img_smpl.img_mask.to(device),
                        smpl_inference=smpl_inference,
                        config=config,
                        num_iters=config["stages"]["reprojection_part"]["num_iters"],
                        verbose="loss" in print_options,
                        iter_fn=save_iter_fn,
                    )
                    reproject_output["pose_body"].append(reproject_output_single["pose_body"])
                    reproject_output["betas"].append(reproject_output_single["betas"])
                    reproject_output["root_orient"].append(reproject_output_single["root_orient"])
                    reproject_output["trans"].append(reproject_output_single["trans"])
                    reproject_output["joints_2d"].append(reproject_output_single["joints_2d"])
                    reproject_output["joints_2d_gt"].append(reproject_output_single["joints_2d_gt"])
                    reproject_output["focal_length"].append(reproject_output_single["focal_length"])
                    reproject_output["reproject_mask"].append(reproject_output_single["reproject_mask"])
                    reproject_output["cam_trans"].append(reproject_output_single["cam_trans"])
                    reproject_output["camera_center"].append(reproject_output_single["camera_center"])
                    reproject_output["input_angle"].append(reproject_output_single["input_angle"])
                    reproject_output["output_angle"].append(reproject_output_single["output_angle"])
                    reproject_output["metrics"].append(reproject_output_single["metrics"])

                if save_iterations:
                    iter_output["reprojection_output"] = {}
                    iter_output["reprojection_output"]["metrics"] = reproject_output["metrics"]
                    iter_output["reprojection_output"]["input_angle"] = reproject_output["input_angle"]
                    iter_output["reprojection_output"]["output_angle"] = reproject_output["output_angle"]

                if config["stages"]["reprojection_part"]["criterion"] == "reprojection":
                    min_reproject_error_index = np.argmin([x["reproject"] for x in reproject_output["metrics"]])
                elif config["stages"]["reprojection_part"]["criterion"] == "chamfer":
                    min_reproject_error_index = np.argmin([x["chamfer"] for x in reproject_output["metrics"]])

                betas = reproject_output["betas"][min_reproject_error_index][0]  # [F, 10]
                o_betas = torch.mean(betas, dim=0, keepdim=True).clone().detach().requires_grad_(False)  # [1, 10]
                root_orient = reproject_output["root_orient"][min_reproject_error_index][0]  # [F, 1, 3, 3]
                o_root_orient = root_orient.clone().detach().requires_grad_(False)  # [F, 1, 3, 3]
                trans = reproject_output["trans"][min_reproject_error_index][0]  # [F, 3]
                o_trans = trans.clone().detach().requires_grad_(False)  # [F, 3]

                # get camera parameters
                cam_trans = reproject_output["cam_trans"][min_reproject_error_index][0]
                cam_trans = cam_trans.clone().detach().requires_grad_(False)  # [F, 3]
                joints_2d_gt = reproject_output["joints_2d_gt"][min_reproject_error_index][0]
                joints_2d_gt = joints_2d_gt.clone().detach().requires_grad_(False)  # [F, 45, 2]
                focal_length = reproject_output["focal_length"][min_reproject_error_index]
                focal_length = focal_length.clone().detach().requires_grad_(False)  # [F, 45, 2]
                camera_center = reproject_output["camera_center"][min_reproject_error_index]
                camera_center = camera_center.clone().detach().requires_grad_(False)  # [F, 2]
                reproject_mask = reproject_output["reproject_mask"][min_reproject_error_index]
                reproject_mask = reproject_mask.clone().detach().requires_grad_(False)  # [F, 2]

            visualize_fn = None
            if visualize_fits:
                visualize_fn=visualize_parts

            filter_output = find_best_part_fits(
                markers=markers,
                pose_body=o_pose_body,
                betas=o_betas,
                root_orient=o_root_orient,
                marker_labels=segmented_markers,
                smpl_inference=smpl_inference,
                hierarchy=smpl_inference.smpl.parents,
                joints_2d_gt=joints_2d_gt,
                focal_length=focal_length,
                reproject_mask=reproject_mask,
                cam_trans=cam_trans,  # [F, 3], requires_grad=False
                camera_center=camera_center,
                config=config,
                foot_contacts=o_foot_contacts,
                visualize_fn=visualize_fn,
                iter_fn=save_iter_fn,
            )
            segmented_markers = filter_output["marker_labels"].detach().clone()
            root_orient = filter_output["root_orient"].detach().clone()
            trans = filter_output["trans"].detach().clone()
            betas = filter_output["betas"].detach().clone()

            smpl_part = {}
            smpl_part["trans"] = trans.clone().detach().cpu().numpy()
            smpl_part["root_orient"] = normalize_rot(root_orient).clone().detach().cpu().numpy()
            smpl_part["betas"] = betas[0].clone().detach().cpu().numpy()
            smpl_part["pose_body"] = normalize_rot(o_pose_body).clone().detach().cpu().numpy()

        marker_labels = segmented_markers.detach().cpu().numpy()

    if not config["find_best_part_fits"] or aabb_volume_ratio > 0.4:
        trans = torch.median(markers, dim=1)[0].requires_grad_(True)
        root_orient = o_root_orient.clone().requires_grad_(True)
        betas = o_betas.clone().requires_grad_(True)

    ########## Stage [reprojection]: find reasonable rotational alignment with  ##########
    if config["stages"]["reprojection_full"]["num_iters"] > 0:
        num_angles = config["stages"]["reprojection_full"]["num_angles"]  # A
        angles = torch.arange(0, 2*np.pi, (2*np.pi)/num_angles)

        reproject_output = {}
        reproject_output["pose_body"] = []
        reproject_output["betas"] = []
        reproject_output["root_orient"] = []
        reproject_output["trans"] = []
        reproject_output["joints_2d"] = []
        reproject_output["joints_2d_gt"] = []
        reproject_output["cam_trans"] = []
        reproject_output["input_angle"] = []
        reproject_output["output_angle"] = []
        reproject_output["metrics"] = []

        for angle_index in range(num_angles):
            angle = angles[angle_index]
            reproject_output_single = optim_reprojection(
                markers=markers,  # [F, M, 3], requires_grad=False
                pose_body=o_pose_body,  # [F, J-1, 3, 3], requires_grad=False
                betas=betas,  # [1, 10], requires_grad=True
                hmr_betas=o_hmr_betas,  # [F, 10], requires_grad=False
                root_orient=o_hmr_root_orient,  # [F, 1, 3, 3], requires_grad=True
                trans=trans,  # [F, 3], requires_grad=True
                pred_cam=o_camera_bbox,  # [F, 3], requires_grad=False
                cam_center=o_center,  # [F, 3], requires_grad=False
                cam_scale=o_scale,  # [F, 3], requires_grad=False
                cam_size=o_size,  # [F, 3], requires_grad=False
                angle=angle,
                smpl_inference=smpl_inference,
                config=config,
                num_iters=config["stages"]["reprojection_part"]["num_iters"],
                verbose="loss" in print_options,
                iter_fn=save_iter_fn,
            )
            reproject_output["pose_body"].append(reproject_output_single["pose_body"])
            reproject_output["betas"].append(reproject_output_single["betas"])
            reproject_output["root_orient"].append(reproject_output_single["root_orient"])
            reproject_output["trans"].append(reproject_output_single["trans"])
            reproject_output["joints_2d"].append(reproject_output_single["joints_2d"])
            reproject_output["joints_2d_gt"].append(reproject_output_single["joints_2d_gt"])
            reproject_output["cam_trans"].append(reproject_output_single["cam_trans"])
            reproject_output["input_angle"].append(reproject_output_single["input_angle"])
            reproject_output["output_angle"].append(reproject_output_single["output_angle"])
            reproject_output["metrics"].append(reproject_output_single["metrics"])

        min_reproject_error_index = np.argmin([x["reproject"] for x in reproject_output["metrics"]])
        betas = reproject_output["betas"][min_reproject_error_index][0]  # [F, 10]
        betas = torch.mean(betas, dim=0, keepdim=True).clone().detach().requires_grad_(True)  # [1, 10]
        root_orient = reproject_output["root_orient"][min_reproject_error_index][0]  # [F, 1, 3, 3]
        root_orient = root_orient.clone().detach().requires_grad_(True)  # [F, 1, 3, 3]
        trans = reproject_output["trans"][min_reproject_error_index][0]  # [F, 3]
        trans = trans.clone().detach().requires_grad_(True)  # [F, 3]

    ########## Stage [root]: optimize rigid transformation to align img smpl and markers ##########
    if "progress" in print_options:
        print("Stage [root]: optimizing root...")

    if config["stages"]["root"]["num_iters"] > 0:
        optim_root(
            markers,
            pose_body=o_pose_body,
            betas=o_betas,
            root_orient=root_orient,
            trans=trans,
            marker_labels=torch.from_numpy(marker_labels).to(device),
            smpl_inference=smpl_inference,
            config=config,
            verbose="loss" in print_options,
            iter_fn=save_iter_fn,
        )
        smpl_root = {}
        smpl_root["trans"] = trans.clone().detach().cpu().numpy()
        smpl_root["root_orient"] = normalize_rot(root_orient).clone().detach().cpu().numpy()
        smpl_root["betas"] = betas[0].clone().detach().cpu().numpy()
        smpl_root["pose_body"] = normalize_rot(o_pose_body).clone().detach().cpu().numpy()

    pose_body = o_pose_body.clone().requires_grad_(True)
    root_orient = root_orient.detach()

    smpl_chamfer_rotations = {}
    smpl_marker_rotations = {}

    root_orient_angles = torch.arange(0, 2*np.pi, (2*np.pi)/config["num_root_orient_angles"]).tolist()
    for root_orient_angle in root_orient_angles:
        root_orient_angle_tensor = torch.tensor([[[root_orient_angle]]]).float().to(device)
        z_root_orient_angle = compute_root_orient_z(torch.repeat_interleave(root_orient_angle_tensor, repeats=root_orient.shape[0], dim=0)) @ root_orient.clone().detach().requires_grad_(True)

        z_root_orient_angle = z_root_orient_angle.clone().detach().requires_grad_(True)
        trans_angle = trans.clone().detach().requires_grad_(True)
        pose_body_angle = pose_body.clone().detach().requires_grad_(True)
        betas_angle = betas.clone().detach().requires_grad_(True)

        smpl_chamfer_rotations[root_orient_angle] = {}
        smpl_marker_rotations[root_orient_angle] = {}

        ########## Stage [pose]: optimize pose to align img smpl and markers ##########
        if True:
            if "progress" in print_options:
                print("Stage [pose]: optimizing poses and shapes...")

            if config["stages"]["chamfer"]["num_iters"] > 0:
                optim_chamfer(
                    markers,
                    pose_body=pose_body_angle,
                    o_pose_body=o_pose_body,
                    betas=betas_angle,
                    o_betas=o_betas,
                    root_orient=z_root_orient_angle,
                    trans=trans_angle,
                    marker_labels=torch.from_numpy(marker_labels).to(device),
                    img_mask=img_smpl.img_mask.to(device),
                    smpl_inference=smpl_inference,
                    initial_angle=root_orient_angle,
                    repeat=0,
                    config=config,
                    verbose="loss" in print_options,
                    iter_fn=save_iter_fn,
                )

            smpl_chamfer_rotations[root_orient_angle] = {}
            smpl_chamfer_rotations[root_orient_angle]["trans"] = trans_angle.clone().detach().cpu().numpy()
            smpl_chamfer_rotations[root_orient_angle]["root_orient"] = normalize_rot(z_root_orient_angle).clone().detach().cpu().numpy()
            smpl_chamfer_rotations[root_orient_angle]["betas"] = betas_angle[0].clone().detach().cpu().numpy()
            smpl_chamfer_rotations[root_orient_angle]["pose_body"] = normalize_rot(pose_body_angle).clone().detach().cpu().numpy()

        ########## Stage: get marker placement ##########
        stage_i = 0
        if True:
            if "progress" in print_options:
                print("Stage: computing marker placement... [{}/{}]".format(stage_i+1, config["stage_repeats"]))

            if config["stages"]["marker"]["num_iters"] > 0:
                barycentric_coords_one_hot_final = compute_nearest_points(
                    markers=markers,
                    pose_body=pose_body_angle,
                    betas=betas_angle,
                    root_orient=z_root_orient_angle,
                    trans=trans_angle,
                    smpl_inference=smpl_inference,
                    marker_labels=marker_labels,
                    granularity=config["stages"]["segment"]["granularity"],
                    img_mask=img_smpl.img_mask.to(device),
                    device=device,
                    config=config,
                    o_pose_body=o_pose_body,
                    window_size=1,
                    use_velocity=config["stages"]["compute_locations"]["use_velocity"],
                )

                if config["recompute_marker_labels"]:
                    marker_labels = compute_marker_labels_from_coords(
                        smpl_inference=smpl_inference,
                        barycentric_coords_one_hot=barycentric_coords_one_hot_final,
                        num_frames=num_frames,
                    ).detach().cpu().numpy()
                    if config["stages"]["segment"]["rigid_filter"]:
                        marker_labels = filter_rigid(
                            markers.detach().cpu().numpy(),
                            marker_labels,
                        )

            ########## Stage [marker]: optimize SMPL parameters using markers ##########
            if "progress" in print_options:
                print("Stage [marker]: optimizing SMPL parameters... [{}/{}]".format(stage_i+1, config["stage_repeats"]))

            if config["stages"]["marker"]["num_iters"] > 0:
                z_root_orient_angle = z_root_orient_angle.clone().detach().requires_grad_(True)
                pose_body_angle = pose_body_angle.clone().detach().requires_grad_(True)

                optim_markers(
                    markers=markers,
                    pose_body=pose_body_angle,
                    o_pose_body=o_pose_body,
                    betas=betas_angle,
                    o_betas=o_betas,
                    root_orient=z_root_orient_angle,
                    trans=trans_angle,
                    barycentric_coords_one_hot=barycentric_coords_one_hot_final,
                    img_mask=img_smpl.img_mask.to(device),
                    smpl_inference=smpl_inference,
                    config=config,
                    initial_angle=root_orient_angle,
                    repeat=0,
                    verbose="loss" in print_options,
                    iter_fn=save_iter_fn,
                )

            z_root_orient_angle = normalize_rot(z_root_orient_angle).clone().detach().requires_grad_(True)
            pose_body_angle = normalize_rot(pose_body_angle).clone().detach().requires_grad_(True)
            
            smpl_marker_rotations[root_orient_angle] = {}
            smpl_marker_rotations[root_orient_angle]["trans"] = trans_angle.clone().detach().cpu().numpy()
            smpl_marker_rotations[root_orient_angle]["root_orient"] = z_root_orient_angle.clone().detach().cpu().numpy()
            smpl_marker_rotations[root_orient_angle]["betas"] = betas_angle[0].clone().detach().cpu().numpy()
            smpl_marker_rotations[root_orient_angle]["pose_body"] = pose_body_angle.clone().detach().cpu().numpy()

    best_angle_chamfer = np.inf
    best_angle = None
    for root_orient_angle in root_orient_angles:
        angle_betas = torch.from_numpy(smpl_marker_rotations[root_orient_angle]["betas"]).to(device)
        angle_betas = torch.repeat_interleave(angle_betas[None], dim=0, repeats=smpl_marker_rotations[root_orient_angle]["pose_body"].shape[0])

        vertices = smpl_inference(
            poses=torch.from_numpy(smpl_marker_rotations[root_orient_angle]["pose_body"]).to(device),  # [F, 23, 3, 3]
            betas=angle_betas,  # [F, 10]
            root_orient=torch.from_numpy(smpl_marker_rotations[root_orient_angle]["root_orient"]).to(device),  # [F, 1, 3, 3]
            trans=torch.from_numpy(smpl_marker_rotations[root_orient_angle]["trans"]).to(device),  # [F, 3]
        )["vertices"]

        chamfer_distance = weighted_chamfer_distance(
            x=markers,
            y=vertices,
            x_weights=get_marker_mask(markers),
            single_directional=True,
        )[0]

        chamfer_distance = chamfer_distance
        if chamfer_distance < best_angle_chamfer:
            best_angle_chamfer = chamfer_distance
            best_angle = root_orient_angle

    smpl_chamfer = smpl_chamfer_rotations[best_angle]
    smpl_marker = smpl_marker_rotations[best_angle]
    root_orient = torch.from_numpy(smpl_marker["root_orient"]).to(device).requires_grad_(True)
    trans = torch.from_numpy(smpl_marker["trans"]).to(device).requires_grad_(True)
    pose_body = torch.from_numpy(smpl_marker["pose_body"]).to(device).requires_grad_(True)
    betas = torch.from_numpy(smpl_marker["betas"][None]).to(device).requires_grad_(True)

    print("Final marker optimization")
    for stage_i in range(config["stage_repeats"]):
        pose_body_stage = torch.clone(pose_body).detach().requires_grad_(False)
        if "progress" in print_options:
            print("Stage: computing marker placement... [{}/{}]".format(stage_i+1, config["stage_repeats"]))

        if config["stages"]["marker"]["num_iters"] > 0:
            barycentric_coords_one_hot_final = compute_nearest_points(
                markers=markers,
                pose_body=pose_body,
                betas=betas,
                root_orient=root_orient,
                trans=trans,
                smpl_inference=smpl_inference,
                marker_labels=marker_labels,
                granularity=config["stages"]["segment"]["granularity"],
                img_mask=img_smpl.img_mask.to(device),
                device=device,
                config=config,
                o_pose_body=pose_body_stage,
                window_size=1,
                use_velocity=config["stages"]["compute_locations"]["use_velocity"],
            )

            if config["recompute_marker_labels"]:
                marker_labels = compute_marker_labels_from_coords(
                    smpl_inference=smpl_inference,
                    barycentric_coords_one_hot=barycentric_coords_one_hot_final,
                    num_frames=num_frames,
                ).detach().cpu().numpy()
                if config["stages"]["segment"]["rigid_filter"]:
                    marker_labels = filter_rigid(
                        markers.detach().cpu().numpy(),
                        marker_labels,
                    )

        ########## Stage [marker]: optimize SMPL parameters using markers ##########
        if "progress" in print_options:
            print("Stage [marker]: optimizing SMPL parameters... [{}/{}]".format(stage_i+1, config["stage_repeats"]))

        if config["stages"]["marker"]["num_iters"] > 0:
            root_orient = root_orient.clone().detach().requires_grad_(True)
            pose_body = pose_body.clone().detach().requires_grad_(True)

            optim_markers(
                markers=markers,
                pose_body=pose_body,
                o_pose_body=pose_body_stage,
                betas=betas,
                o_betas=o_betas,
                root_orient=root_orient,
                trans=trans,
                barycentric_coords_one_hot=barycentric_coords_one_hot_final,
                img_mask=img_smpl.img_mask.to(device),
                smpl_inference=smpl_inference,
                config=config,
                initial_angle=0,
                repeat=1,
                verbose="loss" in print_options,
                iter_fn=save_iter_fn,
            )

        root_orient = normalize_rot(root_orient).clone().detach().requires_grad_(True)
        pose_body = normalize_rot(pose_body).clone().detach().requires_grad_(True)
        
        smpl_marker_final = {}
        smpl_marker_final["trans"] = trans.clone().detach().cpu().numpy()
        smpl_marker_final["root_orient"] = root_orient.clone().detach().cpu().numpy()
        smpl_marker_final["betas"] = betas[0].clone().detach().cpu().numpy()
        smpl_marker_final["pose_body"] = pose_body.clone().detach().cpu().numpy()

    ########## return output ##########
    output = {}
    output["trans"] = trans.detach().cpu()
    output["root_orient"] = normalize_rot(root_orient).detach().cpu()
    output["pose_body"] = normalize_rot(pose_body).detach().cpu()
    output["betas"] = torch.repeat_interleave(torch.mean(betas, dim=0, keepdim=True), dim=0, repeats=pose_body.shape[0]).detach().cpu()
    output["mocap_frame_rate"] = mocap_markers.get_frequency()

    mocap_markers.set_points(markers.detach().cpu().numpy())
    output["mocap_markers"] = mocap_markers
    output["markers_labels"] = marker_labels

    if save_stages:
        output["stages"] = {}
        if config["stages"]["root"]["num_iters"] > 0:
            output["stages"]["root"] = smpl_root
        if config["find_best_part_fits"]:
            output["stages"]["part"] = smpl_part
        if config["stages"]["chamfer"]["num_iters"] > 0:
            output["stages"]["chamfer"] = smpl_chamfer
        if config["stages"]["marker"]["num_iters"] > 0:
            output["stages"]["marker"] = smpl_marker
        if config["stage_repeats"] > 0:
            output["stages"]["marker_final"] = smpl_marker_final

    if filter_output is not None:
        output["chain"] = filter_output["chain"]

    if save_iterations:
        output["iterations"] = iter_output

    return output


def pad(sequence, offset):
    if offset > 0:
        padding = sequence[[0]]
    elif offset < 0:
        padding = sequence[[-1]]
    else:
        return sequence

    padding = torch.repeat_interleave(padding, repeats=abs(offset), dim=0)

    if offset < 0:
        return torch.cat((sequence, padding), dim=0)
    else:
        return torch.cat((padding, sequence), dim=0)
