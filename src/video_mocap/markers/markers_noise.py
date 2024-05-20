import random

import numpy as np


def markers_swap(
    markers,  # [F, M, 3]
    distance_threshold=0.0,
    min_frames=0,
    max_frames=0,
    p=0.0,
):
    num_frames, num_markers, _ = markers.shape

    output = np.array(markers)

    num_swaps = 0

    for frame in range(num_frames):
        for m_i_0 in range(num_markers):
            for m_i_1 in range(m_i_0+1, num_markers):
                if np.linalg.norm((markers[frame, m_i_0] - markers[frame, m_i_1])) < distance_threshold:
                    prob = np.random.uniform(low=0.0, high=1.0)

                    if min_frames < max_frames:
                        block = np.random.randint(min_frames, max_frames)
                    else:
                        block = max_frames

                    if prob < p:
                        output[frame:frame+block, [m_i_0, m_i_1]] = output[frame:frame+block, [m_i_1, m_i_0]]
                        num_swaps += 1

    print("Num marker swaps:", num_swaps)

    return output


def markers_tracking_loss(
    markers,  # [F, M, 3]
    min_frames=0,
    max_frames=0,
    p=0.0,
):
    num_frames, num_markers, _ = markers.shape

    output = np.array(markers)

    num_losses = 0

    for frame in range(num_frames):
        for m_i_0 in range(num_markers):
            prob = np.random.uniform(low=0.0, high=1.0)

            if min_frames < max_frames:
                block = np.random.randint(min_frames, max_frames)
            else:
                block = max_frames

            if prob < p:
                output[frame:frame+block, m_i_0] *= 0
                num_losses += 1

    print("Num marker tracking losses:", num_losses)

    return output


def markers_tracking_loss_second_block(
    markers,  # [F, M, 3]
    window_size,
    p=0.0,
):
    num_frames, num_markers, _ = markers.shape

    output = np.array(markers)

    num_losses = 0

    for frame in range(0, num_frames, window_size):
        indices = random.sample(range(num_markers), k=int(p*num_markers))
        for index in indices:
            output[frame:frame+window_size, index] = 0.0

    print("Num marker tracking losses:", num_losses)

    return output
