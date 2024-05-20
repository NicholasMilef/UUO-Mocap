import torch
import torch.nn as nn
import torch.nn.functional as F


class FootContactModel(nn.Module):
    def __init__(
        self,
        num_parts = 24,  # number of parts
        latent_dim=128,
        sequence_length=32,
    ):
        super().__init__()

        self.num_parts = num_parts
        self.sequence_length = sequence_length

        self.j_embedding = nn.Linear(self.num_parts * 3, latent_dim)
        self.j_conv0a = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
        self.j_conv1a = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
        self.j_conv2a = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))

        self.fc_a = nn.Linear(latent_dim, latent_dim)
        self.fc_b = nn.Linear(latent_dim, 2)


    def forward(
        self,
        img_smpl_joints,  # [N, F, 22, 3]
    ):
        batch_size, num_frames, num_joints, _ = img_smpl_joints.shape
        img_smpl_joints_reshape = torch.reshape(img_smpl_joints, (batch_size, num_frames, 1, num_joints * 3))  # [N, F, 1, 66]
        j_embed = self.j_embedding(img_smpl_joints_reshape)  # [N, F, 1, 128]
        j_x_embed = torch.permute(j_embed, (0, 3, 2, 1))  # [N, 128, M, F]
        j_x0 = F.relu(self.j_conv0a(j_x_embed))  # [N, 128, M, F]
        #j_x0 = F.relu(self.j_conv0b(j_x0))  # [N, 128, M, F]
        j_x0 = F.max_pool2d(j_x0, kernel_size=(1, 4))  # [N, 128, M, F/4]
        j_x1 = F.relu(self.j_conv1a(j_x0))  # [N, 128, M, F/4]
        #j_x1 = F.relu(self.j_conv1b(j_x1))  # [N, 128, M, F/4]
        j_x1 = F.max_pool2d(j_x1, kernel_size=(1, 4))  # [N, 128, M, F/16]
        j_x2 = F.relu(self.j_conv2a(j_x1))  # [N, 128, M, F/16]
        #j_x2 = F.relu(self.j_conv2b(j_x2))  # [N, 128, M, F/16]
        j_x2 = F.max_pool2d(j_x2, kernel_size=(1, 2))  # [N, 128, M, F/32]
        j_x2 = torch.permute(j_x2, (0, 2, 1, 3))  # [N, M, 128, F/32]
        j_x2 = torch.flatten(j_x2, start_dim=2, end_dim=3)  # [N, 128, M, F/32] -> [N, 1, 128]

        out = self.fc_a(j_x2)
        out = self.fc_b(out)

        return out
