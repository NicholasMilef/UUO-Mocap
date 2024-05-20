import os

import numpy as np
from pytorch3d.transforms import matrix_to_axis_angle, matrix_to_quaternion, quaternion_to_matrix
import torch
import roma

from video_mocap.utils.img_smpl_utils import get_foot_contacts
from video_mocap.utils.img_smpl_utils import joints_3d_labels


class ImgSmpl:
    def __init__(self, data, freq):
        self.data = data
        self.freq = freq
    
        num_frames = len(self.data)

        smpl_data = {
            "trans": np.zeros((num_frames, 3), dtype=np.float32),
            "root_orient": np.zeros((num_frames, 1, 3, 3), dtype=np.float32),
            "hmr_root_orient": np.zeros((num_frames, 1, 3, 3), dtype=np.float32),
            "pose_body": np.zeros((num_frames, 23, 3, 3), dtype=np.float32),
            "betas": np.zeros((num_frames, 10), dtype=np.float32),
        }
        self.camera_bbox = np.zeros((num_frames, 3), dtype=np.float32)  # [F, 3]
        self.center = np.zeros((num_frames, 2), dtype=np.float32)  # [F, 2]
        self.size = np.zeros((num_frames, 2), dtype=np.float32)  # [F, 2]
        self.scale = np.zeros((num_frames, 1), dtype=np.float32)  # [F, 2]

        self.img_mask = torch.zeros((num_frames)).bool()

        # process SMPL data
        frame_index = 0
        for key in sorted(self.data.keys()):  # for each frame
            if len(self.data[key]["tracked_ids"]) > 0:
                self.img_mask[frame_index] = 1.0
                root_orient = self.data[key]["smpl"][0]["global_orient"]
                correction_matrix = np.array([
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0],
                ]).astype(np.float32)
                smpl_data["hmr_root_orient"][frame_index] = root_orient
                smpl_data["trans"][frame_index] = self.data[key]["3d_joints"][0][joints_3d_labels["pelvis_low"]]
                smpl_data["root_orient"][frame_index] = correction_matrix @ root_orient
                smpl_data["pose_body"][frame_index] = self.data[key]["smpl"][0]["body_pose"]
                smpl_data["betas"][frame_index] = self.data[key]["smpl"][0]["betas"]
            frame_index += 1

        for key, _ in smpl_data.items():
            smpl_data[key] = torch.from_numpy(np.stack(smpl_data[key], axis=0))

        # fill in missing data
        valid_frames = torch.where(self.img_mask == True)[0]
        for frame in range(num_frames):
            if frame not in valid_frames.tolist():
                left = valid_frames[valid_frames < frame]
                right = valid_frames[valid_frames > frame]

                if left.size()[0] == 0:
                    right = right[0]
                    smpl_data["hmr_root_orient"][frame] = smpl_data["hmr_root_orient"][right]
                    smpl_data["trans"][frame] = smpl_data["trans"][right]
                    smpl_data["root_orient"][frame] = smpl_data["root_orient"][right]
                    smpl_data["pose_body"][frame] = smpl_data["pose_body"][right]
                    smpl_data["betas"][frame] = smpl_data["betas"][right]
                elif right.size()[0] == 0:
                    left = left[-1]
                    smpl_data["hmr_root_orient"][frame] = smpl_data["hmr_root_orient"][left]
                    smpl_data["trans"][frame] = smpl_data["trans"][left]
                    smpl_data["root_orient"][frame] = smpl_data["root_orient"][left]
                    smpl_data["pose_body"][frame] = smpl_data["pose_body"][left]
                    smpl_data["betas"][frame] = smpl_data["betas"][left]
                else:
                    left = left[-1]
                    right = right[0]
                    alpha = ((frame - left) / (right - left)).item()
                    
                    smpl_data["trans"][frame] = (smpl_data["trans"][left] * (1.0 - alpha)) + (smpl_data["trans"][right] * (alpha))
                    smpl_data["betas"][frame] = (smpl_data["betas"][left] * (1.0 - alpha)) + (smpl_data["betas"][right] * (alpha))
                    
                    alpha = torch.tensor([alpha])
                    smpl_data["hmr_root_orient"][frame] = quaternion_to_matrix(roma.utils.unitquat_slerp(
                        matrix_to_quaternion(smpl_data["hmr_root_orient"][left]),
                        matrix_to_quaternion(smpl_data["hmr_root_orient"][right]),
                        alpha,
                    ))
                    smpl_data["root_orient"][frame] = quaternion_to_matrix(roma.utils.unitquat_slerp(
                        matrix_to_quaternion(smpl_data["root_orient"][left]),
                        matrix_to_quaternion(smpl_data["root_orient"][right]),
                        alpha,
                    ))
                    smpl_data["pose_body"][frame] = quaternion_to_matrix(roma.utils.unitquat_slerp(
                        matrix_to_quaternion(smpl_data["pose_body"][left]),
                        matrix_to_quaternion(smpl_data["pose_body"][right]),
                        alpha,
                    ))

        self.trans = smpl_data["trans"]
        self.root_orient = smpl_data["root_orient"]
        self.hmr_root_orient = smpl_data["hmr_root_orient"]
        self.pose_body = smpl_data["pose_body"]
        self.betas = smpl_data["betas"]

        # process camera data
        frame_index = 0
        for key in sorted(self.data.keys()):  # for each frame
            if len(self.data[key]["camera_bbox"]) > 0:
                self.camera_bbox[frame_index] = self.data[key]["camera_bbox"][0]
                self.center[frame_index] = self.data[key]["center"][0]
                self.scale[frame_index] = self.data[key]["scale"][0]
                self.size[frame_index] = self.data[key]["size"][0]
            frame_index += 1

        self.camera_bbox = torch.from_numpy(self.camera_bbox)
        self.center = torch.from_numpy(self.center)
        self.scale = torch.from_numpy(self.scale)
        self.size = torch.from_numpy(self.size)

        # process 2D joints
        joints_2d = np.zeros((len(data), 45, 2))
        frame = 0
        for key in sorted(list(data.keys())):
            for i in range(45):
                try:
                    joints_2d[frame, i] = data[key]["2d_joints"][0][i*2:i*2+2]
                except:
                    pass
            frame += 1

        self.foot_contacts = torch.from_numpy(get_foot_contacts(joints_2d, freq).astype(np.float32))

    def get_smpl(self):
        poses = torch.cat((self.root_orient, self.pose_body), dim=1)  # [F, J, 3, 3]
        poses = matrix_to_axis_angle(poses)  # [F, J, 3]
        poses = torch.flatten(poses, start_dim=1, end_dim=-1)  # [F, J*3]

        output = {}
        output["betas"] = self.betas[0].detach().cpu().numpy()  # [10]
        output["gender"] = np.array("neutral")
        output["mocap_frame_rate"] = self.freq
        output["poses"] = poses.detach().cpu().numpy()  # [F, J*3]
        output["trans"] = self.trans.detach().cpu().numpy()  # [F, 3]
        return output
