import itertools
from typing import List

import numpy as np
import torch

from moshpp.marker_layout.marker_vids import all_marker_vids
from smplx.joint_names import SMPL_JOINT_NAMES


joint_colors = {
    "pelvis":           np.array([0.0, 0.0, 1.0]),  # 0
    "left_hip":         np.array([1.0, 0.0, 1.0]),  # 1
    "right_hip":        np.array([1.0, 1.0, 0.0]),  # 2
    "spine1":           np.array([0.3, 0.3, 1.0]),  # 3
    "left_knee":        np.array([0.7, 0.0, 0.7]),  # 4
    "right_knee":       np.array([0.7, 0.7, 0.0]),  # 5
    "spine2":           np.array([0.5, 0.5, 1.0]),  # 6
    "left_ankle":       np.array([0.5, 0.0, 0.5]),  # 7
    "right_ankle":      np.array([0.5, 0.5, 0.0]),  # 8
    "spine3":           np.array([0.7, 0.7, 1.0]),  # 9
    "left_foot":        np.array([0.2, 0.0, 0.2]),  # 10
    "right_foot":       np.array([0.2, 0.2, 0.0]),  # 11
    "neck":             np.array([1.0, 0.4, 0.0]),  # 12
    "left_collar":      np.array([1.0, 0.2, 0.0]),  # 13
    "right_collar":     np.array([0.2, 0.4, 0.0]),  # 14
    "head":             np.array([0.5, 0.2, 0.0]),  # 15
    "left_shoulder":    np.array([0.2, 0.0, 0.0]),  # 16
    "right_shoulder":   np.array([0.0, 0.2, 0.0]),  # 17
    "left_elbow":       np.array([0.5, 0.0, 0.0]),  # 18
    "right_elbow":      np.array([0.0, 0.5, 0.0]),  # 19
    "left_wrist":       np.array([0.8, 0.0, 0.0]),  # 20
    "right_wrist":      np.array([0.0, 0.8, 0.0]),  # 21
    "left_hand":        np.array([1.0, 0.0, 0.0]),  # 22
    "right_hand":       np.array([0.0, 1.0, 0.0]),  # 23
}


def get_joint_id(joint_name):
    return SMPL_JOINT_NAMES.index(joint_name)


def get_joint_name(joint_id):
    return SMPL_JOINT_NAMES[joint_id]


def get_all_joint_ids():
    return [get_joint_id(x) for x in SMPL_JOINT_NAMES]


def get_joint_colors(id):
    return joint_colors[SMPL_JOINT_NAMES[id]]


def get_joint_colors_vertices(vertices):
    colors = np.zeros((vertices.shape[0], 3))
    for i in range(vertices.shape[0]):
        joint_id = vertices[i]
        colors[i] = get_joint_colors(joint_id)
    return colors


def get_marker_vertex_id(marker_name, model_type="smpl"):
    return all_marker_vids[model_type][marker_name]


smpl_limbs = {
    "head": [get_joint_id("head")],
    "left_arm": [get_joint_id("left_shoulder"), get_joint_id("left_elbow"), get_joint_id("left_wrist"), get_joint_id("left_hand")],
    "left_leg": [get_joint_id("left_hip"), get_joint_id("left_knee"), get_joint_id("left_foot"), get_joint_id("left_ankle")],
    "left_shoulder": [get_joint_id("left_collar"), get_joint_id("left_shoulder"), get_joint_id("left_elbow")],
    "right_arm": [get_joint_id("right_shoulder"), get_joint_id("right_elbow"), get_joint_id("right_wrist"), get_joint_id("right_hand")],
    "right_leg": [get_joint_id("right_hip"), get_joint_id("right_knee"), get_joint_id("right_foot"), get_joint_id("right_ankle")],
    "right_shoulder": [get_joint_id("right_collar"), get_joint_id("right_shoulder"), get_joint_id("right_elbow")],
}

smpl_not_limbs = [
    get_joint_id("pelvis"),
    get_joint_id("left_hip"),
    get_joint_id("right_hip"),
    get_joint_id("spine1"),
    get_joint_id("spine2"),
    get_joint_id("spine3"),
    get_joint_id("neck"),
    get_joint_id("left_collar"),
    get_joint_id("right_collar"),
    get_joint_id("head"),
    get_joint_id("left_shoulder"),
    get_joint_id("right_shoulder"),
]


smpl_joint_symmetry = [
    [get_joint_id("left_hip"), get_joint_id("right_hip")],
    [get_joint_id("left_knee"), get_joint_id("right_knee")],
    [get_joint_id("left_ankle"), get_joint_id("right_ankle")],
    [get_joint_id("left_foot"), get_joint_id("right_foot")],
    [get_joint_id("left_collar"), get_joint_id("right_collar")],
    [get_joint_id("left_shoulder"), get_joint_id("right_shoulder")],
    [get_joint_id("left_elbow"), get_joint_id("right_elbow")],
    [get_joint_id("left_wrist"), get_joint_id("right_wrist")],
    [get_joint_id("left_hand"), get_joint_id("right_hand")],
]


def get_sub_hierachies(
    parents: np.ndarray|torch.Tensor,
    num_bones: int,
) -> List[List[int]]:
    """
    Get sub-hierarchies containing number of bones

    Args:
        parents: 
        num_bones: number of bones

    Returns:
        List[List[int]]
    """
    output = []
    children = {}

    # test to make sure the number of bones is not larger than the hierarchy
    if num_bones > len(parents):
        num_bones = len(parents)

    if isinstance(parents, np.ndarray):
        parents_np = parents
    else:
        parents_np = parents.detach().cpu().numpy()

    # initialize data structure
    for i in range(0, parents_np.shape[0]):
        children[i] = []

    for i in range(1, parents_np.shape[0]):
        parent = int(parents_np[i])
        children[parent].append(i)

    # create subtree lookup table
    subtrees_table = {}
    def store_subtrees(node):
        subtrees_table[node] = [[]]  # add empty subtree

        product = itertools.product(*[subtrees_table[x] for x in children[node]])
        for subtree in product:
            subtree_combined = []
            for subtree_i in subtree:
                subtree_combined = subtree_combined + subtree_i
            subtree_combined = sorted(subtree_combined)
            if [node] + subtree_combined not in subtrees_table[node]:
                subtrees_table[node].append([node] + subtree_combined)

    for node in list(children.keys())[::-1]:
        store_subtrees(node)

    # find subtrees of length [1 < length < num_bones]
    subtrees_list = []
    for node, subtrees in subtrees_table.items():
        for subtree in subtrees:
            if len(subtree) == num_bones:
                subtrees_list.append(subtree)

    return subtrees_list


def remove_approximately_redundant_hierarchies(
    subtrees_list: List[List[int]],
    similarity_threshold: float=0.9,
):
    output = [subtrees_list[0]]
    
    for subtree_i in range(1, len(subtrees_list)):
        subtree = subtrees_list[subtree_i]
        num_similarity_nodes = len(subtree) * similarity_threshold

        add_subtree = True
        for output_subtree in output:
            num_common = len(list(set(subtree) & set(output_subtree)))
            if num_common > num_similarity_nodes:
                add_subtree = False

        if add_subtree:
            output.append(subtree)

    print("Retained", str(len(output)) + "/" + str(len(subtrees_list)), "elements")

    return output


if __name__ == "__main__":
    joint_name = "pelvis"
    joint_id = get_joint_id(joint_name)

    print("ID:", get_joint_id(joint_name))
    print("Color:", get_joint_colors(joint_id))
