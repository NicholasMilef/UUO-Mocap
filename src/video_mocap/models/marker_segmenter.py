import torch
import torch.nn as nn
import torch.nn.functional as F


class MarkerSegmenter(nn.Module):
    def __init__(
        self,
        num_parts=24,  # number of joints
        latent_dim=128,
        sequence_length=32,
    ):
        super().__init__()

        self.num_parts = num_parts
        self.sequence_length = sequence_length

        # marker embedding
        self.m_embedding = nn.Linear(3, latent_dim)
        self.m_conv0a = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
        self.m_conv0b = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
        self.m_conv1a = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
        self.m_conv1b = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
        self.m_conv2a = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
        self.m_conv2b = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))

        # root orient embedding
        self.ro_embedding = nn.Linear(3, latent_dim)
        self.ro_conv0a = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
        self.ro_conv0b = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
        self.ro_conv1a = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
        self.ro_conv1b = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
        self.ro_conv2a = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))
        self.ro_conv2b = nn.Conv2d(latent_dim, latent_dim, kernel_size=(1, 3), padding=(0, 1))

        final_latent_dim = ((sequence_length // 32) * latent_dim) * 2
        if "root_orient" in self.extra_features:
            final_latent_dim = ((sequence_length // 32) * latent_dim) * 3

        self.segment_a = nn.Linear(final_latent_dim, final_latent_dim)
        self.segment_b = nn.Linear(final_latent_dim, num_parts)


    def forward(
        self,
        marker_pos,  # [N, F, M, 3]
        root_orient=None,  # [N, F, 3]
    ):
        use_root_orient = True
        if root_orient is not None and "root_orient" not in self.extra_features:
            raise Warning("root_orient is ignored because it has not been specified as an extra_feature in the constructor")
        if root_orient is None or "root_orient" not in self.extra_features:
            use_root_orient = False

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

        global_features_expanded = torch.repeat_interleave(
            global_features,
            repeats=m_embed.shape[2],
            dim=1,
        )

        # root orient embedding
        if use_root_orient:
            ro_embed = self.ro_embedding(root_orient)  # [N, F, 128]
            ro_x_embed = torch.permute(torch.unsqueeze(ro_embed, 2), (0, 3, 2, 1))  # [N, 128, M, F]
            ro_x0 = self.ro_conv0a(ro_x_embed)  # [N, 128, M, F]
            ro_x0 = F.relu(self.ro_conv0b(ro_x0))  # [N, 128, M, F]
            ro_x0 = F.max_pool2d(ro_x0, kernel_size=(1, 4))  # [N, 128, M, F/4]
            ro_x1 = self.ro_conv1a(ro_x0)  # [N, 128, M, F/4]
            ro_x1 = F.relu(self.ro_conv1b(ro_x1))  # [N, 128, M, F/4]
            ro_x1 = F.max_pool2d(ro_x1, kernel_size=(1, 4))  # [N, 128, M, F/16]
            ro_x2 = self.ro_conv2a(ro_x1)  # [N, 128, M, F/16]
            ro_x2 = F.relu(self.ro_conv2b(ro_x2))  # [N, 128, M, F/16]
            ro_x2 = F.max_pool2d(ro_x2, kernel_size=(1, 2))  # [N, 128, M, F/32]
            ro_x2 = torch.permute(ro_x2, (0, 2, 1, 3))  # [N, M, 128, F/32]
            ro_x2 = torch.flatten(ro_x2, start_dim=2, end_dim=3)  # [N, 128, M, F/32]
            ro_x2 = torch.repeat_interleave(ro_x2, repeats=m_x2.shape[1], dim=1)

        if use_root_orient:
            out = F.relu(self.segment_a(torch.cat((global_features_expanded, m_x2, ro_x2), dim=-1)))
        else:
            out = F.relu(self.segment_a(torch.cat((global_features_expanded, m_x2), dim=-1)))


        out = self.segment_b(out)

        return out


    def forward_sequence(
        self,
        marker_pos,  # [N, F, M, 3]
        stride: int=4,
    ):
        if marker_pos.shape[1] // stride < self.sequence_length:
            raise ValueError("sequence length must be at least " + str(self.sequence_length * stride) + " frames")

        batch_size, num_frames, num_markers, _ = marker_pos.shape

        output = torch.zeros((batch_size, num_frames, num_markers, self.num_parts)).to(marker_pos.device)

        total_stride = stride * self.sequence_length
        for i in range(0, num_frames, total_stride):
            sub_seq = marker_pos[:, i:i+total_stride:stride]

            if sub_seq.shape[1] < self.sequence_length:
                continue

            pred_labels = torch.unsqueeze(self.forward(sub_seq), dim=1)  # [N, M, P]
            pred_labels = torch.repeat_interleave(pred_labels, dim=1, repeats=total_stride)
            output[:, i:i+total_stride] = pred_labels

        return output
