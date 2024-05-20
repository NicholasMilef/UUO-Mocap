import numpy as np

# generated via http://vrl.cs.brown.edu/color
# steps for generating something similar (randomized output)
# 1. Set "Number of colors" to 24
# 2. Set "Perceptual Distance" to maximum
# 3. Set "Pair Preference", "Name Difference", and "Name Uniqueness" to minimum
# 4. Set "Lightness Range" to [15, 95]
colors_perceptually_distinct_24 = np.array([
    [6, 150, 104],
    [237, 94, 147],
    [142, 253, 78],
    [150, 32, 252],
    [149, 184, 51],
    [120, 20, 134],
    [227, 250, 172],
    [79, 32, 56],
    [139, 243, 244],
    [166, 0, 62],
    [31, 183, 0],
    [238, 128, 254],
    [37, 80, 38],
    [223, 187, 227],
    [104, 60, 0],
    [61, 255, 196],
    [254, 43, 28],
    [69, 141, 166],
    [227, 112, 16],
    [89, 121, 254],
    [232, 253, 24],
    [1, 54, 136],
    [246, 189, 83],
    [163, 117, 158],
]) / 255.0