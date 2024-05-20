import numpy as np


joints_2d_labels = {
    "nose":             0,
    "neck_high":        1,
    "r_shoulder_low":   2,
    "r_elbow_in":       3,
    "r_wrist_low":      4,
    "l_shoulder_low":   5,
    "l_elbow_in":       6,
    "l_wrist_low":      7,
    "pelvis_low":       8,
    "r_hip":            9,
    "r_knee_low":       10,
    "r_ankle_low":      11,
    "l_hip":            12,
    "l_knee_low":       13,
    "l_ankle_low":      14,
    "r_eye":            15,
    "l_eye":            16,
    "r_ear":            17,
    "l_ear":            18,
    "l_toe_in":         19,
    "l_toe_out":        20,
    "l_heel":           21,
    "r_toe_in":         22,
    "r_toe_out":        23,
    "r_heel":           24,
    "r_ankle_high":     25,
    "r_knee_high":      26,
    "r_pelvis":         27,
    "l_pelvis":         28,
    "l_knee_high":      29,
    "l_ankle_high":     30,
    "r_wrist_high":     31,
    "r_elbow_out":      32,
    "r_shoulder_high":  33,
    "l_shouler_high":   34,
    "l_elbow_out":      35,
    "l_wrist_high":     36,
    "neck_low":         37,
    "c_head_low":       38,
    "pelvis_high":      39,
    "chest_high":       40,
    "chest_low":        41,
    "mouth":            41,
    "c_head_high":      43,
    "c_hip":            44,
}

joints_3d_labels = joints_2d_labels

def get_foot_contacts(joints_2d, freq):
    min_x = np.ones((joints_2d.shape[0])) * np.inf
    max_x = np.ones((joints_2d.shape[0])) * -np.inf
    min_y = np.ones((joints_2d.shape[0])) * np.inf
    max_y = np.ones((joints_2d.shape[0])) * -np.inf
    
    min_x = np.min(joints_2d[:, :, 0], axis=1)
    max_x = np.max(joints_2d[:, :, 0], axis=1)
    min_y = np.min(joints_2d[:, :, 1], axis=1)
    max_y = np.max(joints_2d[:, :, 1], axis=1)
    
    threshold = 0.0001
    epsilon = 0.01

    extent = np.sqrt(((max_x - min_x)**2) + ((max_y - min_y)**2))
    extent = np.maximum(extent, np.ones_like(extent) * epsilon)
    threshold = threshold / extent
    threshold = np.repeat(np.expand_dims(threshold, axis=-1), repeats=joints_2d.shape[1], axis=-1)

    joints_vel_2d = joints_2d[1:] - joints_2d[:-1]
    joints_vel_2d = np.concatenate((np.zeros((1, 45, 2)), joints_vel_2d), axis=0)
    joints_vel_2d = (joints_vel_2d / freq)  # [F, J, 2]
    joints_speed_2d = np.linalg.norm(joints_vel_2d, axis=-1)

    foot_contact = (joints_speed_2d < threshold)  # [F, J]
    toe_indices = [
        [joints_2d_labels["l_toe_in"], joints_2d_labels["l_toe_out"]],
        [joints_2d_labels["r_toe_in"], joints_2d_labels["r_toe_out"]],
    ]

    output = np.ones((joints_2d.shape[0], len(toe_indices)))

    group = 0
    for toe_index_group in toe_indices:
        for toe_index in toe_index_group:
            output[:, group] *= foot_contact[:, toe_index]
        group += 1

    return output