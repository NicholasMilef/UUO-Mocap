"""
Utilities for different datasets
"""
from typing import Dict


def get_camera_name(
    dataset: str,
) -> str:
    """
    Gets the camera name for each dataset. Some datasets have multiple camera
    views with different camera names, so a single view must be selected.

    Args:
        dataset: name of dataset

    Returns:
        str: camera name
    """
    camera_name = {
        "umpm": "l",
        "cmu_kitchen_pilot": "7151062",
        "cmu_kitchen_pilot_rb": "7151062",
        "moyo_train": None,
        "moyo_val": None,
        "bmlmovi_train": None,
        "bmlmovi_val": None,
    }[dataset]
    return camera_name


def get_overlay_font(
    dataset: str,
) -> Dict:
    """
    When overlaying text, some dataset have brighter/darker backgrounds. To
    help with visibility, this function will return appropriate text colors
    and sizes.
    """
    font = {
        "umpm": {"color": (0, 0, 0), "scale": 3.0, "pos": (20, 50)},
        "moyo_val": {"color": (255, 255, 255), "scale": 2, "pos": (20, 50)},
    }[dataset]
    return font