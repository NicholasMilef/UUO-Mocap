import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from video_mocap.utils.random_utils import set_random_seed


class MarkerSegmenterMultimodal(nn.Module):
    def __init__(
        self,
        num_parts=24,  # number of joints
        latent_dim=128,
        sequence_length=32,
        modalities=["markers", "video"],
    ):
        super().__init__()

        self.num_parts = num_parts
        self.sequence_length = sequence_length
        self.modalities = modalities

        if "markers" not in self.modalities:
            raise Warning("Modalities must include \"markers\"")

        # marker embedding
        self.m_embedding = nn.Linear(3, latent_dim)
        self.m_conv0a = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
        #self.m_conv0b = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
        self.m_conv1a = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
        #self.m_conv1b = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
        self.m_conv2a = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
        #self.m_conv2b = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))

        # joint embedding
        if "video" in self.modalities:
            self.j_embedding = nn.Linear(self.num_parts * 3, latent_dim)
            self.j_conv0a = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
            self.j_conv0b = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
            self.j_conv1a = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
            self.j_conv1b = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
            self.j_conv2a = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
            self.j_conv2b = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))

        if "video" in self.modalities:
            final_latent_dim = ((sequence_length // 32) * latent_dim) * 3
        else:
            final_latent_dim = ((sequence_length // 32) * latent_dim) * 2

        self.segment_a = nn.Linear(final_latent_dim, final_latent_dim)
        self.segment_b = nn.Linear(final_latent_dim, num_parts)


    def forward(
        self,
        marker_pos,  # [N, F, M, 3]
        img_smpl_joints,  # [N, F, 22, 3]
    ):        
        # marker embedding
        m_embed = self.m_embedding(marker_pos)  # [N, F, M, 128]
        m_x_embed = torch.permute(m_embed, (0, 3, 2, 1))  # [N, 128, M, F]
        m_x0 = F.relu(self.m_conv0a(m_x_embed))  # [N, 128, M, F]
        #m_x0 = F.relu(self.m_conv0b(m_x0))  # [N, 128, M, F]
        m_x0 = F.max_pool2d(m_x0, kernel_size=(1, 4))  # [N, 128, M, F/4]
        m_x1 = F.relu(self.m_conv1a(m_x0))  # [N, 128, M, F/4]
        #m_x1 = F.relu(self.m_conv1b(m_x1))  # [N, 128, M, F/4]
        m_x1 = F.max_pool2d(m_x1, kernel_size=(1, 4))  # [N, 128, M, F/16]
        m_x2 = F.relu(self.m_conv2a(m_x1))  # [N, 128, M, F/16]
        #m_x2 = F.relu(self.m_conv2b(m_x2))  # [N, 128, M, F/16]
        m_x2 = F.max_pool2d(m_x2, kernel_size=(1, 2))  # [N, 128, M, F/32]
        m_x2 = torch.permute(m_x2, (0, 2, 1, 3))  # [N, M, 128, F/32]
        m_x2 = torch.flatten(m_x2, start_dim=2, end_dim=3)  # [N, 128, M, F/32]

        #global_features = torch.sum(m_x2, dim=1, keepdim=True) # [N, 1, 128]
        global_features = torch.max_pool2d(m_x2, kernel_size=(m_x2.shape[1], 1)) # [N, 1, 128]

        global_features_expanded = torch.repeat_interleave(
            global_features,
            repeats=m_embed.shape[2],
            dim=1,
        )

        # joint embedding
        if "video" in self.modalities:
            batch_size, num_frames, num_joints, _ = img_smpl_joints.shape
            img_smpl_joints_reshape = torch.reshape(img_smpl_joints, (batch_size, num_frames, 1, num_joints * 3))  # [N, F, 1, 66]
            j_embed = self.j_embedding(img_smpl_joints_reshape)  # [N, F, 1, 128]
            j_x_embed = torch.permute(j_embed, (0, 3, 2, 1))  # [N, 128, M, F]
            j_x0 = F.relu(self.j_conv0a(j_x_embed))  # [N, 128, M, F]
            j_x0 = F.relu(self.j_conv0b(j_x0))  # [N, 128, M, F]
            j_x0 = F.max_pool2d(j_x0, kernel_size=(1, 4))  # [N, 128, M, F/4]
            j_x1 = F.relu(self.j_conv1a(j_x0))  # [N, 128, M, F/4]
            j_x1 = F.relu(self.j_conv1b(j_x1))  # [N, 128, M, F/4]
            j_x1 = F.max_pool2d(j_x1, kernel_size=(1, 4))  # [N, 128, M, F/16]
            j_x2 = F.relu(self.j_conv2a(j_x1))  # [N, 128, M, F/16]
            j_x2 = F.relu(self.j_conv2b(j_x2))  # [N, 128, M, F/16]
            j_x2 = F.max_pool2d(j_x2, kernel_size=(1, 2))  # [N, 128, M, F/32]
            j_x2 = torch.permute(j_x2, (0, 2, 1, 3))  # [N, M, 128, F/32]
            j_x2 = torch.flatten(j_x2, start_dim=2, end_dim=3)  # [N, 128, M, F/32] -> [N, 1, 128]

            joint_features_expanded = torch.repeat_interleave(
                j_x2,
                repeats=m_embed.shape[2],
                dim=1,
            )

        if "video" in self.modalities:
            out = F.relu(self.segment_a(F.relu(torch.cat((global_features_expanded, m_x2, joint_features_expanded), dim=-1))))
        else:
            out = F.relu(self.segment_a(F.relu(torch.cat((global_features_expanded, m_x2), dim=-1))))

        out = self.segment_b(out)

        return out


    def forward_sequence(
        self,
        marker_pos,  # [N, F, M, 3]
        img_smpl_joints,  # [N, F, 22, 3]
        stride: int=4,
        center=True,  # center markers
        seed=None,
    ):
        if marker_pos.shape[1] // stride < self.sequence_length:
            warnings.warn("Padding because sequence length is not at least " + str(self.sequence_length * stride) + " frames")

        o_num_frames = marker_pos.shape[1]
        
        total_stride = stride * self.sequence_length

        if center:
            median = torch.median(marker_pos[:, :, :2], dim=1, keepdim=True).values  # [N, 2]
            marker_pos_temp = marker_pos.clone()
            marker_pos_temp[:, :, :2] = marker_pos_temp[:, :, :2] - median
            marker_pos = marker_pos_temp

        # add padding
        pad_size = total_stride - (marker_pos.shape[1] % total_stride)
        marker_pos_padding = torch.repeat_interleave(marker_pos[:, [-1]], repeats=pad_size, dim=1)
        img_smpl_joints_padding = torch.repeat_interleave(img_smpl_joints[:, [-1]], repeats=pad_size, dim=1)

        marker_pos = torch.cat((marker_pos, marker_pos_padding), dim=1)
        img_smpl_joints = torch.cat((img_smpl_joints, img_smpl_joints_padding), dim=1)

        # create output tensor
        batch_size, num_frames, num_markers, _ = marker_pos.shape
        output = torch.zeros((batch_size, num_frames, num_markers, self.num_parts)).to(marker_pos.device)

        for i in range(0, num_frames, total_stride):
            sub_seq_markers = marker_pos[:, i:i+total_stride:stride]
            sub_seq_joints = img_smpl_joints[:, i:i+total_stride:stride]

            if seed is not None:
                set_random_seed(seed)

            pred_labels = torch.unsqueeze(self.forward(sub_seq_markers, sub_seq_joints), dim=1)  # [N, M, P]
            pred_labels = torch.repeat_interleave(pred_labels, dim=1, repeats=total_stride)
            output[:, i:i+total_stride] = pred_labels

        return output[:, :o_num_frames]
