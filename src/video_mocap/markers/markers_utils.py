from collections.abc import Callable
import os
import shutil
from typing import List, Dict

import numpy as np
import pybullet
import pybullet_data
from pytorch3d.loss.chamfer import chamfer_distance
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d, so3_relative_angle
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from sklearn.cluster import AgglomerativeClustering
import torch
import torch.nn.functional as F
from tqdm import tqdm

from video_mocap.models.marker_segmenter import MarkerSegmenter
from video_mocap.optimization import chamfer_distance_by_part, compute_root_orient_y, compute_root_orient_z
from video_mocap.utils.aabb import get_aabb, get_aabb_volume
from video_mocap.utils.hmr_utils import apply_matrix_33_to_vector_3, convert_mocap_pos_to_hmr_pos, convert_hmr_pos_to_mocap_pos, perspective_projection
from video_mocap.utils.settings import MARKER_DISTANCE
from video_mocap.utils.smpl_utils import get_joint_id, get_joint_name, get_sub_hierachies, remove_approximately_redundant_hierarchies


def shuffle_markers(points):
    output = np.zeros_like(points)
    for f in range(points.shape[0]):
        permutation = np.random.permutation(points.shape[1])
        output[f] = np.ascontiguousarray(points[f, permutation])
    return output


def segment_markers(markers, device):
    with torch.no_grad():
        sequence_length = 32
        stride = 4
        target_freq = 30
        num_classes = 52

        joint_segmenter = MarkerSegmenter(
            latent_dim=64,
        )

        joint_segmenter_filename = os.path.join("./checkpoints/marker_segmenter/final/model.pth")
        joint_segmenter.load_state_dict(
            torch.load(joint_segmenter_filename, map_location=device),
        )
        joint_segmenter.to(device)

        points = markers.get_points()
        points = np.nan_to_num(points, 0)

        points_input = torch.from_numpy(points.astype(np.float32)).to(device)
        
        temporal_stride = markers.get_frequency() // target_freq

        num_frames, num_markers, _ = points_input.shape

        classes = torch.zeros((num_frames, num_markers, num_classes)).to(device)

        full_stride = stride * temporal_stride
            
        points_input = points_input.unsqueeze(0)

        for f in range(0, len(markers), full_stride * sequence_length):
            start_frame = f
            end_frame = f + (full_stride * sequence_length)

            window = points_input[:, start_frame:end_frame:full_stride]  # [1, F, M, 3]
            window = pad_window(window, sequence_length)
            segmentation = torch.softmax(joint_segmenter(window), dim=-1)            

            classes[start_frame:end_frame] = segmentation

        return classes


def pad_window(window, sequence_length):
    current_length = window.shape[1]

    if current_length == sequence_length:
        return window

    end_frame = window[:, [-1]]
    padding = torch.repeat_interleave(
        end_frame,
        repeats=sequence_length-current_length,
        dim=1,
    )
    return torch.cat((window, padding), dim=1)


def id_markers(
    points: np.ndarray,
) -> np.ndarray:
    """
    Label markers from point cloud

    Args:
        points: [F, M, 3]

    Returns:
        np.ndarray: [F, M, 3]
    """
    output = np.zeros_like(points)
    output[0] = points[0]

    for f in range(1, points.shape[0]):
        matching = np.zeros((points.shape[1], points.shape[1]))
        for m_i in range(points.shape[1]):
            for m_j in range(points.shape[1]):
                matching[m_i, m_j] = np.linalg.norm(
                    output[f-1, m_i] - points[f, m_j]
                )
        _, order_1 = min_weight_full_bipartite_matching(csr_matrix(matching))
        output[f] = np.ascontiguousarray(points[f, order_1])

    return output


def randomly_drop_markers(
    points: np.ndarray,
    frequency: float,
    marker_radius: float = 0.01,
    num_drop: int=0,
) -> np.ndarray:
    """
    Args:
        points: [F, M, 3]
        frequency: mocap frequency
        marker_radius: marker radius in meters
        num_drop: number of markers to drop
        
    Returns:
        np.ndarray: [F, M, 3]
    """
    if num_drop == 0:
        return points

    client = pybullet.connect(pybullet.DIRECT)
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    pybullet.setGravity(0, 0, -9.8)

    drop_frames = [0] * num_drop
    drop_indices = np.random.permutation(points.shape[1])[:num_drop]

    drop_frame_offset = 0
    for i in range(num_drop):
        drop_frame_offset += points.shape[0] // (num_drop + 1)
        drop_frames[i] = drop_frame_offset

    # create
    floor_collision_shape = pybullet.createCollisionShape(
        pybullet.GEOM_BOX,
        halfExtents=[10, 10, 0.5],
    )
    floor_body = pybullet.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=floor_collision_shape,
        basePosition=[0, 0, -0.5],
    )

    markers = []

    output = np.array(points)
    for f in range(points.shape[0]):
        for i in range((240 // int(frequency)) - 1):
            pybullet.stepSimulation()
        pybullet.stepSimulation()

        for m_i in range(num_drop):
            marker_index = drop_indices[m_i]
            if f == drop_frames[m_i]:
                marker_collision_shape = pybullet.createCollisionShape(
                    pybullet.GEOM_SPHERE,
                    radius=marker_radius,
                )
                markers.append(pybullet.createMultiBody(
                    baseMass=0.05,
                    baseCollisionShapeIndex=marker_collision_shape,
                    basePosition=points[f, marker_index],
                ))
                linear_vel = points[f, marker_index] - points[f-1, marker_index]
                pybullet.resetBaseVelocity(markers[-1], linearVelocity=linear_vel)

            if f >= drop_frames[m_i]:
                pos, rot = pybullet.getBasePositionAndOrientation(markers[m_i])
                output[f, marker_index] = pos

    pybullet.disconnect()

    return output


def cleanup_markers(
    points: np.ndarray,
) -> np.ndarray:
    """
    Cleans up invalid markers. Note that some markers will be removed through this process.
    
    Args:
        points: [F, M, 3]

    Returns:
        np.ndarray: [F, M_c, 3]
    """
    output = []
    for m in range(points.shape[1]):
        m_pos = points[:, m]
        m_vel = m_pos[1:] - m_pos[:-1]
        m_speed = np.linalg.norm(m_vel, axis=-1)
        if np.median(m_speed, axis=0) > 0:
            output.append(m_pos)

    output = np.stack(output, axis=1)
    return output


def filter_rigid(
    points: np.ndarray,
    labels: np.ndarray,
) -> List[List[int]]:
    """
    Filter labels using rigid segmentation
    
    Args:
        points: [F, M, 3]
        labels: [F, M]

    Returns:
        Dict: list of clusters, with list of marker IDs
    """
    rigid_segmentation = segment_rigid(points)

    output = np.array(labels)
    for group in rigid_segmentation:
        group_labels = np.median(labels[:, group])
        output[:, group] = group_labels

    return output


def segment_rigid(
    points: np.ndarray,  # [F, M, 3]
) -> List[List[int]]:
    """
    Segments points into rigid bodies
    
    Args:
        points: [F, M, 3]

    Returns:
        List[List[int]]: list of clusters, with list of marker IDs
    """
    _, num_markers, _ = points.shape
    mat_A = np.zeros((num_markers, num_markers))
    for i in range(num_markers):
        for j in range(num_markers):
            distance = np.linalg.norm(points[:, i] - points[:, j], axis=-1)
            mat_A[i, j] = np.std(distance)

    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.005, metric="precomputed", linkage="average")
    clustering_output = clustering.fit(mat_A)

    output = []
    for value in np.unique(clustering_output.labels_).tolist():
        indices = np.where(clustering_output.labels_ == value)[0].tolist()
        output.append(indices)

    return output


def find_best_part_fits(
    markers: torch.Tensor,  # [F, M, 3]
    pose_body: torch.Tensor,  # [F, J-1, 3, 3]
    betas: torch.Tensor,  # [1, 10]
    root_orient: torch.Tensor,  # [F, 1, 3, 3]
    marker_labels: torch.Tensor,  # [F, M]
    smpl_inference,
    hierarchy: torch.Tensor,  # [J]
    joints_2d_gt: torch.Tensor,  # [F, 45, 2], requires_grad=False
    focal_length: torch.Tensor,  # [F, 45, 2], requires_grad=False
    reproject_mask: torch.Tensor,
    camera_center: torch.Tensor,  # [F, 3], requires_grad=False
    cam_trans: torch.Tensor,  # [F, 3], requires_grad=False
    config: Dict,
    foot_contacts: torch.Tensor=None,  # [F, 2]
    visualize_fn = None,
    iter_fn: Callable=None,
):
    device = markers.device

    labels_mode = torch.mode(marker_labels, axis=0)[0]  # [M]
    label_groups = torch.unique(labels_mode, return_counts=True)
    label_groups = [x.tolist() for x in label_groups]  # convert to lists

    final_marker_labels = torch.zeros_like(marker_labels)
    final_marker_weights = torch.zeros_like(marker_labels, dtype=torch.float)

    num_frames, num_markers, _ = markers.shape

    if config["stages"]["part"]["mode"] == "network":
        # merge left-right chains (doesn't matter because of multiple hypothesis testing)
        label_groups_map = {}
        for i in range(len(label_groups[0])):
            joint = label_groups[0][i]
            joint_name = get_joint_name(joint)
            num_joint_markers = label_groups[1][i]

            new_joint_name = joint_name.replace("right", "left")
            new_joint_id = get_joint_id(new_joint_name)

            if new_joint_id != joint:
                labels_mode[labels_mode==joint] = new_joint_id

            if new_joint_id not in label_groups_map:
                label_groups_map[new_joint_id] = 0
            label_groups_map[new_joint_id] += num_joint_markers

        label_groups = [
            list(label_groups_map.keys()),
            list(label_groups_map.values()),
        ]

        # chains are actually subtrees
        # find unique kinematic chains
        chains = []
        for i in range(hierarchy.shape[0]):
            if i in label_groups[0]:
                not_found = True
                for chain in chains:
                    if hierarchy[i] in chain:
                        chain.append(i)
                        not_found = False
                        break

                if not_found:
                    chains.append([i])
    elif config["stages"]["part"]["mode"] == "cluster":
        chains = [label_groups[0]]

    o_betas = betas

    final_betas = None
    final_distance = np.inf
    final_markers_subset = None
    final_root_orient = None
    final_trans = None

    # take only largest chain
    largest_chain = chains[0]
    largest_chain_num_markers = 0
    for chain in chains:
        num_markers_per_chain = 0
        for joint in chain:
            num_markers_per_chain += torch.where(labels_mode==joint)[0].shape[0]

        if config["stages"]["part"]["mode"] == "network":
            print("Found:", str([get_joint_name(x) for x in chain]) + ":", num_markers_per_chain)
        elif config["stages"]["part"]["mode"] == "cluster":
            print("Found sequence with length", str(len(chain)))

        if len(chain) >= len(largest_chain) and num_markers_per_chain > largest_chain_num_markers:
            largest_chain = chain
            largest_chain_num_markers = num_markers_per_chain
    chains = [largest_chain]

    # fit limb for each subtree (chain)
    if visualize_fn is not None:
        if os.path.exists("render_output"):
            shutil.rmtree("render_output")
        os.makedirs("render_output", exist_ok=True)

    for i in range(len(chains)):
        chain = chains[i]

        marker_labels_subset_list = []

        indices = []
        for joint in chain:
            indices.append(torch.where(labels_mode==joint)[0])
        indices = torch.cat(indices, dim=0)

        if config["stages"]["part"]["mode"] == "network":
            print([get_joint_name(x) for x in chain], len(indices))

        for joint in chain:
            marker_labels_subset_list.append([joint, (marker_labels==joint)[:, indices]])

        markers_subset = markers[:, indices]
        markers_subset_mean = torch.mean(markers_subset, dim=1)

        if "use_full_skeleton" in config["stages"]["part"] and config["stages"]["part"]["use_full_skeleton"]:
            subtrees = [np.arange(0, hierarchy.shape[0]).tolist()]
        else:
            subtrees = get_sub_hierachies(hierarchy, len(chain))

            if "similarity_threshold" in config["stages"]["part"]:
                subtrees = remove_approximately_redundant_hierarchies(
                    subtrees,
                    similarity_threshold=0.9,
                )

        if "reproject" in config["stages"]["part"]["losses"]:
            correction_matrix = torch.tensor([[[
                [1, 0, 0],
                [0, 0, 1],
                [0, -1, 0],
            ]]]).to(device).float()  # [1, 1, 3, 3]
            correction_matrix = torch.repeat_interleave(correction_matrix, repeats=num_frames, dim=0)

        # process each part
        progress = tqdm(total=len(subtrees))
        subtree_losses = []
        for subtree in subtrees:
            z_angle = torch.zeros((1, 1, 1)).to(device).requires_grad_(True)  # [1, 1, 1]
            trans = torch.median(markers, dim=1)[0].clone().requires_grad_(True)
            betas = o_betas.clone().requires_grad_(True)

            # if reprojection, camera translation must be solved for
            if "reproject" in config["stages"]["part"]["losses"]:
                cam_trans_single = cam_trans[[0]].clone()  # [1, 3]
                params = [z_angle] + [trans] + [betas] + [cam_trans_single]
            else:
                params = [z_angle] + [trans] + [betas]

            optimizer = torch.optim.LBFGS(
                params,
                max_iter=config["stages"]["part"]["num_iters"],
                tolerance_grad=config["optimizer"]["tolerance_grad"],
                tolerance_change=config["optimizer"]["tolerance_change"],
                lr=1.0,
                line_search_fn="strong_wolfe",
            )

            vertex_indices_list = []

            weights = smpl_inference.get_lbs_weights()
            vertex_labels = torch.argmax(weights, dim=-1)

            for joint in subtree:
                vertex_indices_list.append((vertex_labels==joint).nonzero(as_tuple=True)[0])

            vertex_indices = torch.cat(vertex_indices_list, dim=0)

            # create new marker matrix
            if not ("use_full_skeleton" in config["stages"]["part"] and config["stages"]["part"]["use_full_skeleton"]):
                subtree_marker_labels = torch.zeros_like(markers_subset[:, :, 0], dtype=torch.int32)
                for j_i in range(len(subtree)):
                    subtree_marker_labels[marker_labels_subset_list[j_i][1]] = subtree[j_i]

            iteration = 0
            def closure_fit_subtree():
                optimizer.zero_grad()
                nonlocal iteration

                z_root_orient = compute_root_orient_z(torch.repeat_interleave(z_angle, repeats=num_frames, dim=0)) @ root_orient

                smpl_output = smpl_inference(
                    poses=pose_body,
                    betas=torch.repeat_interleave(betas, dim=0, repeats=pose_body.shape[0]),
                    root_orient=z_root_orient,
                    trans=trans,
                )

                loss = 0

                vertices_subset = smpl_output["vertices"][:, vertex_indices]

                loss_chamfer = chamfer_distance(
                    markers_subset,
                    vertices_subset,
                    single_directional=True,
                )[0]
                loss += loss_chamfer * config["stages"]["part"]["losses"]["chamfer"]

                #loss += F.mse_loss(z_root_orient_vel, root_orient_vel) * 1.0

                if "reproject" in config["stages"]["part"]["losses"]:
                    # convert camera translation and orientation to HMR space
                    hmr_cam_trans = convert_mocap_pos_to_hmr_pos(cam_trans_single)
                    hmr_cam_trans = torch.repeat_interleave(hmr_cam_trans, dim=0, repeats=num_frames)  # [F, 3]
                    hmr_root_orient = torch.linalg.inv(correction_matrix) @ root_orient

                    # get camera offset in HMR space
                    pred_smpl_translations = convert_mocap_pos_to_hmr_pos(trans).requires_grad_()
                    camera_offset = pred_smpl_translations - hmr_cam_trans  # [F, 3]

                    # apply rotation to offset
                    inv_translation = apply_matrix_33_to_vector_3(
                        compute_root_orient_y(z_angle)[:, 0],
                        camera_offset,
                    )  # [F, 3]
                    inv_translation = inv_translation + hmr_cam_trans  # [F, 3]

                    smpl_output_hmr = smpl_inference(
                        poses=pose_body,  # [F, J-1, 3, 3]
                        betas=torch.repeat_interleave(betas, dim=0, repeats=pose_body.shape[0]),  # [F, 10]
                        root_orient=hmr_root_orient,  # [F, 1, 3, 3]
                        trans=inv_translation,  # [F, 3]
                    )
                    pred_keypoints_2d_smpl = perspective_projection(
                        points=smpl_output_hmr["joints"],  # [F, 45, 3]
                        translation=hmr_cam_trans,  # [F, 3]
                        focal_length=torch.repeat_interleave(focal_length, dim=0, repeats=1*num_frames),  # [A*F, 3]
                        camera_center=torch.repeat_interleave(camera_center, dim=0, repeats=1),  # [A*F, 2]
                        rotation=torch.eye(3,).unsqueeze(0).expand(1*num_frames, -1, -1).to(device),  # [A*F, 3, 3]
                    ).reshape((num_frames, 45, 2)) + 0.5  # [F, 45, 2]

                    reproject_loss = torch.mean((pred_keypoints_2d_smpl - joints_2d_gt)**2 * reproject_mask[:, None, None])
                    loss += reproject_loss * config["stages"]["part"]["losses"]["reproject"]

                # beta regularization
                if "reg_betas" in config["stages"]["part"]["losses"]:
                    loss += F.mse_loss(betas, o_betas) * config["stages"]["part"]["losses"]["reg_betas"]

                # foot contact height regularization
                if "foot_contact" in config["stages"]["part"]["losses"] and foot_contacts is not None:
                    feet_height = smpl_output["joints"][:, [get_joint_id("left_foot"), get_joint_id("right_foot")], 2]
                    foot_contact_loss = F.mse_loss(feet_height, torch.ones_like(feet_height) * 0.005, reduction="none")
                    loss += torch.mean(foot_contact_loss * foot_contacts) * config["stages"]["part"]["losses"]["foot_contact"]

                # foot velocity regularization
                if "foot_velocity" in config["stages"]["part"]["losses"] and foot_contacts is not None:
                    foot_velocity_xy = smpl_output["joints"][1:, [get_joint_id("left_foot"), get_joint_id("right_foot")], :2]\
                          - smpl_output["joints"][:-1, [get_joint_id("left_foot"), get_joint_id("right_foot")], :2]
                    foot_speed_xy = torch.norm(foot_velocity_xy, dim=-1)
                    foot_speed_loss = F.mse_loss(foot_speed_xy, torch.zeros_like(foot_speed_xy), reduction="none") * config["stages"]["part"]["losses"]["foot_velocity"]
                    loss += torch.mean(foot_speed_loss * foot_contacts[1:]) * 1.0

                # translation velocity regularizer
                if "velocity" in config["stages"]["part"]["losses"]:
                    trans_vel = trans[1:] - trans[:-1]
                    markers_subset_vel = markers_subset_mean[1:] - markers_subset_mean[:-1]
                    loss += F.mse_loss(trans_vel, markers_subset_vel) * config["stages"]["part"]["losses"]["velocity"]

                # prevent ground penetration
                if "ground" in config["stages"]["part"]["losses"]:
                    ground_loss = torch.mean(F.relu(-smpl_output["vertices"][..., 2]))
                    #ground_loss = torch.mean(F.relu(-torch.min(smpl_output["vertices"][..., 2])))
                    loss += ground_loss * config["stages"]["part"]["losses"]["ground"]

                loss.backward()

                if iter_fn is not None:
                    part_name = ", ".join([get_joint_name(x) for x in subtree])
                    iter_fn(
                        stage="part",
                        iteration=iteration,
                        pose_body=pose_body.detach().cpu().numpy(),
                        betas=betas.detach().cpu().numpy(),
                        trans=trans.detach().cpu().numpy(),
                        root_orient=z_root_orient.detach().cpu().numpy(),
                        markers=markers_subset.detach().cpu().numpy(),
                        part=part_name,
                        part_joints=np.array(subtree),
                    )

                iteration += 1
                #print(loss.item())
                return loss

            optimizer.step(closure_fit_subtree)

            z_root_orient = compute_root_orient_z(torch.repeat_interleave(z_angle, repeats=num_frames, dim=0)) @ root_orient
            smpl_output = smpl_inference(
                poses=pose_body,
                betas=torch.repeat_interleave(betas, dim=0, repeats=pose_body.shape[0]),
                root_orient=z_root_orient,
                trans=trans,
            )
            vertices_subset = smpl_output["vertices"][:, vertex_indices]

            distance = chamfer_distance(
                markers_subset,
                vertices_subset,
                single_directional=False,
            )[0].item()
            subtree_losses.append([subtree, distance])

            if distance < final_distance:
                final_betas = betas.clone()
                final_distance = distance
                final_markers_subset = markers_subset.clone()  # [F, M_s, 3]
                final_root_orient = z_root_orient.clone()
                final_trans = trans.clone()
                aabb_volume_ratio = get_aabb_volume(get_aabb(markers_subset)) / get_aabb_volume(get_aabb(markers))

                vertex_labels_subset = vertex_labels[vertex_indices]
                #TODO: get marker label selection working
                for i in range(indices.shape[0]):
                    m_i = indices[i]
                    marker_distance = torch.norm(smpl_output["vertices"] - markers_subset[:, [i]], dim=-1)
                    mean_marker_distance = torch.mean(marker_distance, dim=0)
                    min_marker_distance = torch.argmin(mean_marker_distance, dim=-1)
                    final_marker_labels[:, m_i] = vertex_labels[min_marker_distance]

            if visualize_fn is not None:
                visualize_fn(
                    subtree,
                    markers,
                    smpl_output["vertices"],
                    torch.from_numpy(smpl_inference.smpl.faces.astype(np.int32)),
                    marker_labels,
                    indices,
                    vertex_indices,
                )
            progress.update(1)
        progress.close()

        if len(subtree_losses) > 1:
            subtree_losses = sorted(subtree_losses, key=lambda x: x[1])
            for i in range(indices.shape[0]):
                m_i = indices[i]
                final_marker_weights[:, m_i] = subtree_losses[1][1] / subtree_losses[0][1]
                if indices.shape[0] == 1:
                    final_marker_weights *= 0

        for loss_index in range(min(len(subtree_losses), 3)):
            joint_names = [get_joint_name(x) for x in subtree_losses[loss_index][0]]
            print(", ".join(joint_names), "{:.6f}".format(subtree_losses[loss_index][1]))
        print("--------------------")

    # rescale marker weights
    final_marker_weights = final_marker_weights / torch.max(final_marker_weights)

    output = {}
    output["betas"] = final_betas.clone()
    output["marker_labels"] = final_marker_labels.clone()
    output["markers_subset"] = final_markers_subset.clone()
    output["marker_weights"] = final_marker_weights.clone()
    output["root_orient"] = final_root_orient.clone()
    output["trans"] = final_trans.clone()
    output["aabb_volume_ratio"] = aabb_volume_ratio.clone()
    output["chain"] = np.array([x for x in subtree_losses[0][0]], dtype=np.int32)

    return output
