import numpy as np
import torch
from scipy.signal import savgol_filter

from video_mocap.utils.smpl_utils import get_joint_id


def compute_foot_contacts_torch(joints: torch.Tensor):
    """
    Compute foot contacts

    Args:
        joints: [N, F, J, 3]

    Returns:
        foot_contacts: [N, F, 2]
    """

    joints_np = joints.detach().cpu().numpy()
    foot_contacts = compute_foot_contacts_np(joints_np)
    return torch.from_numpy(foot_contacts).to(joints.device)


def compute_foot_contacts_np(joints: np.ndarray):
    """
    Compute foot contacts

    Args:
        joints: [N, F, J, 3]

    Returns:
        foot_contacts: [N, F, 2]
    """
    batch_size, num_frames, _, _ = joints.shape

    window_size = 7

    left_foot = joints[:, :, [get_joint_id("left_foot")], :]
    right_foot = joints[:, :, [get_joint_id("right_foot")], :]

    # position mask
    left_foot_min = np.percentile(left_foot[..., 2], 10)
    right_foot_min = np.percentile(right_foot[..., 2], 10)
    floor_height = min(left_foot_min, right_foot_min)
    height_threshold = 0.05  # 5 cm
    l_height_mask = (left_foot[..., 1] <= floor_height + height_threshold).astype(float)
    r_height_mask = (right_foot[..., 1] <= floor_height + height_threshold).astype(float)

    # velocity mask
    pad_vel = np.zeros((1, 1, 1, 3))
    lf_vel = np.concatenate((pad_vel, np.diff(left_foot, axis=1)), axis=1)
    rf_vel = np.concatenate((pad_vel, np.diff(right_foot, axis=1)), axis=1)
    try:
        lf_speed = savgol_filter(np.linalg.norm(lf_vel, axis=-1), window_size, 3, axis=1)
        rf_speed = savgol_filter(np.linalg.norm(rf_vel, axis=-1), window_size, 3, axis=1)
    except:
        import pdb; pdb.set_trace()
    vel_threshold = 0.005  # 5 mm/frame
    l_vel_mask = (lf_speed <= vel_threshold).astype(float)
    r_vel_mask = (rf_speed <= vel_threshold).astype(float)

    # full mask
    l_mask = l_height_mask * l_vel_mask
    r_mask = r_height_mask * r_vel_mask

    foot_contacts = np.stack((l_mask, r_mask), axis=-1)[:, :, 0, :]

    return foot_contacts