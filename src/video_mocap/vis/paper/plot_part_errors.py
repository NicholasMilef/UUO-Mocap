import argparse
import os
import csv

import matplotlib.pyplot as plt
import numpy as np


filenames = [
    ("arm", "results/stats/umpm/left_arm/video_mocap.csv", "left"),
    ("arm", "results/stats/umpm/right_arm/video_mocap.csv", "right"),
    ("leg", "results/stats/umpm/left_leg/video_mocap.csv", "left"),
    ("leg", "results/stats/umpm/right_leg/video_mocap.csv", "right"),
    ("shoulder", "results/stats/umpm/left_shoulder/video_mocap.csv", "left"),
    ("shoulder", "results/stats/umpm/right_shoulder/video_mocap.csv", "right"),
]

m2s = {"arm": np.zeros((12, 2)), "leg": np.zeros((12, 2)), "shoulder": np.zeros((12, 2))}
mpjpe = {"arm": np.zeros((12, 2)), "leg": np.zeros((12, 2)), "shoulder": np.zeros((12, 2))}
mpjve = {"arm": np.zeros((12, 2)), "leg": np.zeros((12, 2)), "shoulder": np.zeros((12, 2))}

for filename in filenames:
    with open(filename[1], "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        row_index = 0
        for row in csv_reader:
            if row_index != 0:
                if filename[2] == "left":
                    m2s[filename[0]][row_index-1, 0] = row[2]
                    mpjpe[filename[0]][row_index-1, 0] = row[3]
                    mpjve[filename[0]][row_index-1, 0] = row[4]
                if filename[2] == "right":
                    m2s[filename[0]][row_index-1, 1] = row[2]
                    mpjpe[filename[0]][row_index-1, 1] = row[3]
                    mpjve[filename[0]][row_index-1, 1] = row[4]          
            row_index += 1

nbins = 10

plt.clf()
fig, ((m2s0, m2s1, m2s2)) = plt.subplots(1, 3, figsize=(5, 2))
m2s0.hist(m2s["arm"], nbins, histtype="bar", stacked=True, label=["left", "right"])
m2s0.legend()
m2s0.set_title("Arms")
m2s0.set_ylabel("m2s (mm)")
m2s1.hist(m2s["leg"], nbins, histtype="bar", stacked=True)
m2s1.set_title("Legs")
m2s2.hist(m2s["shoulder"], nbins, histtype="bar", stacked=True)
m2s2.set_title("Shoulders")
fig.tight_layout()
#plt.show()
plt.savefig("paper/part_error_m2s.pdf")

plt.clf()
fig, ((mpjpe0, mpjpe1, mpjpe2)) = plt.subplots(1, 3, figsize=(5, 2))
mpjpe0.hist(mpjpe["arm"], nbins, histtype="bar", stacked=True, label=["left", "right"])
#mpjpe0.legend()
mpjpe0.set_title("Arms")
mpjpe0.set_ylabel("MPJPE (mm)")
mpjpe1.hist(mpjpe["leg"], nbins, histtype="bar", stacked=True)
mpjpe1.set_title("Legs")
mpjpe2.hist(mpjpe["shoulder"], nbins, histtype="bar", stacked=True)
mpjpe2.set_title("Shoulders")
fig.tight_layout()
#plt.show()
plt.savefig("paper/part_error_mpjpe.pdf")

