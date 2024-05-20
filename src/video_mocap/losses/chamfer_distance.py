from pytorch3d.loss import chamfer_distance
import torch


def weighted_chamfer_distance(
    x: torch.Tensor,  # [N, P1, D]
    y: torch.Tensor,  # [N, P2, D]
    x_weights: torch.Tensor,  # [N, P1]
    single_directional: bool = False,
):
    x_flattened = torch.reshape(x, (-1, 1, x.shape[2]))
    y_flattened = torch.repeat_interleave(y, repeats=x.shape[1], dim=0)
    x_weights_flattened = torch.flatten(x_weights)

    distance = chamfer_distance(
        x_flattened,
        y_flattened,
        weights=x_weights_flattened,
        single_directional=True,
    )
    return distance


if __name__ == "__main__":
    batch_size = 16
    p_1 = 7
    p_2 = 19
    dim = 3

    x = torch.range(0, batch_size * p_1 * dim - 1).reshape(batch_size, p_1, dim)
    y = torch.range(0, batch_size * p_2 * dim - 1).reshape(batch_size, p_2, dim)
    x_weights = torch.ones(batch_size, p_1)
    x_weights[:, ::2] = 0

    loss, _ = weighted_chamfer_distance(x, y, x_weights)
    print(loss.item())
