import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarkerEmbedding(nn.Module):
    def __init__(
        self,
        output_dim=32,
        latent_dim=128,
        sequence_length=32,
    ):
        super().__init__()

        # marker embedding
        self.m_embedding = nn.Linear(3, latent_dim)
        self.m_conv0a = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
        self.m_conv0b = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
        self.m_conv1a = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
        self.m_conv1b = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
        self.m_conv2a = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
        self.m_conv2b = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))

        final_latent_dim = ((sequence_length // 32) * latent_dim)

        self.embed_markers = nn.Linear(final_latent_dim, output_dim)

    def forward(
        self,
        marker_pos,  # [N, F, M, 3]
    ):
        # marker embedding
        m_embed = self.m_embedding(marker_pos)  # [N, F, M, 128]
        m_x_embed = torch.permute(m_embed, (0, 3, 2, 1))  # [N, 128, M, F]
        m_x0 = self.m_conv0a(m_x_embed)  # [N, 128, M, F]
        m_x0 = F.relu(self.m_conv0b(m_x0))  # [N, 128, M, F]
        m_x0 = F.max_pool2d(m_x0, kernel_size=(1, 4))  # [N, 128, M, F/4]
        m_x1 = self.m_conv1a(m_x0)  # [N, 128, M, F/4]
        m_x1 = F.relu(self.m_conv1b(m_x1))  # [N, 128, M, F/4]
        m_x1 = F.max_pool2d(m_x1, kernel_size=(1, 4))  # [N, 128, M, F/16]
        m_x2 = self.m_conv2a(m_x1)  # [N, 128, M, F/16]
        m_x2 = F.relu(self.m_conv2b(m_x2))  # [N, 128, M, F/16]
        m_x2 = F.max_pool2d(m_x2, kernel_size=(1, 2))  # [N, 128, M, F/32]
        m_x2 = torch.permute(m_x2, (0, 2, 1, 3))  # [N, M, 128, F/32]
        m_x2 = torch.flatten(m_x2, start_dim=2, end_dim=3)  # [N, 128, M, F/32]

        global_features = torch.sum(m_x2, dim=1, keepdim=True) # [N, 1, 128]

        out = F.normalize(self.embed_markers(F.relu(global_features)), dim=-1)

        return out


class JointEmbedding(nn.Module):
    def __init__(
        self,
        output_dim=32,
        latent_dim=128,
        sequence_length=32,
    ):
        super().__init__()

        self.sequence_length = sequence_length

        # joint embedding
        self.j_embedding = nn.Linear(66, latent_dim)
        self.j_conv0a = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
        self.j_conv0b = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
        self.j_conv1a = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
        self.j_conv1b = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
        self.j_conv2a = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
        self.j_conv2b = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))

        final_latent_dim = ((sequence_length // 32) * latent_dim)

        self.embed_markers = nn.Linear(final_latent_dim, output_dim)

    def forward(
        self,
        img_smpl_joints,  # [N, F, 22, 3]
    ):
        # joint embedding
        batch_size, num_frames, num_joints, _ = img_smpl_joints.shape
        img_smpl_joints_reshape = torch.reshape(img_smpl_joints, (batch_size, num_frames, 1, num_joints * 3))  # [N, F, 1, 66]
        j_embed = self.j_embedding(img_smpl_joints_reshape)  # [N, F, 1, 128]
        j_x_embed = torch.permute(j_embed, (0, 3, 2, 1))  # [N, 128, M, F]
        j_x0 = self.j_conv0a(j_x_embed)  # [N, 128, M, F]
        j_x0 = F.relu(self.j_conv0b(j_x0))  # [N, 128, M, F]
        j_x0 = F.max_pool2d(j_x0, kernel_size=(1, 4))  # [N, 128, M, F/4]
        j_x1 = self.j_conv1a(j_x0)  # [N, 128, M, F/4]
        j_x1 = F.relu(self.j_conv1b(j_x1))  # [N, 128, M, F/4]
        j_x1 = F.max_pool2d(j_x1, kernel_size=(1, 4))  # [N, 128, M, F/16]
        j_x2 = self.j_conv2a(j_x1)  # [N, 128, M, F/16]
        j_x2 = F.relu(self.j_conv2b(j_x2))  # [N, 128, M, F/16]
        j_x2 = F.max_pool2d(j_x2, kernel_size=(1, 2))  # [N, 128, M, F/32]
        j_x2 = torch.permute(j_x2, (0, 2, 1, 3))  # [N, M, 128, F/32]
        j_x2 = torch.flatten(j_x2, start_dim=2, end_dim=3)  # [N, 128, M, F/32] -> [N, 1, 128]

        out = F.normalize(self.embed_markers(F.relu(j_x2)), dim=-1)

        return out


class TemporalAlignmentModel:
    def __init__(
        self,
        output_dim = 32,
        latent_dim = 128,
        sequence_length=32,
        stride=4,
        device=torch.device("cpu"),
    ):
        self.marker_embedding = MarkerEmbedding(
            output_dim=output_dim,
            sequence_length=sequence_length,
            latent_dim=latent_dim,
        )
        self.marker_embedding.load_state_dict(torch.load(
            "./checkpoints/motion_embedding/10_2_test/marker_embedding.pth",
            map_location=device,
        ))
        self.marker_embedding.to(device)

        self.joint_embedding = JointEmbedding(
            output_dim=output_dim,
            sequence_length=sequence_length,
            latent_dim=latent_dim,
        )
        self.joint_embedding.load_state_dict(torch.load(
            "./checkpoints/motion_embedding/10_2_test/joint_embedding.pth",
            map_location=device,
        ))
        self.joint_embedding.to(device)

        self.sequence_length = sequence_length
        self.stride = stride
        self.device = device

    def compute_offset(
        self,
        marker_pos,  # [N, F, M, 3]
        img_smpl_joints,  # [N, F, J, 3]
        temp_stride,
    ):
        full_stride = temp_stride * self.stride
        window_size = self.sequence_length * self.stride

        # downsample to 30Hz
        marker_pos_processed = marker_pos[:, ::temp_stride]
        joints_processed = img_smpl_joints[:, ::temp_stride]
        num_frames_downsample = marker_pos_processed.shape[1]

        # pad
        def pad_sequence(seq, pad_size):
            num_pad_frames = pad_size - seq.shape[1]
            padding = torch.zeros((seq.shape[0], num_pad_frames, seq.shape[2], seq.shape[3])).float().to(self.device)
            output = torch.cat((seq, padding), dim=1)
            return output

        min_pad = max(
            num_frames_downsample + marker_pos_processed.shape[1],
            num_frames_downsample + window_size,
        )
        marker_pos_processed = pad_sequence(marker_pos_processed, min_pad)
        joints_processed = pad_sequence(joints_processed, min_pad)

        matrix = torch.zeros(num_frames_downsample, num_frames_downsample).float().to(self.device)

        for i in range(0, num_frames_downsample):
            for j in range(0, num_frames_downsample):
                marker_embedding = self.marker_embedding(marker_pos_processed[:, i:i+window_size:self.stride])
                joints_embedding = self.joint_embedding(joints_processed[:, j:j+window_size:self.stride])
                matrix[i, j] = torch.norm(marker_embedding[0, 0] - joints_embedding[0, 0])

        return full_stride
