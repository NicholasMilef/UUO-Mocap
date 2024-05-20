"""
Produces qualitative results
If rendering on a server, add the following text preceeding the python command:
DISPLAY=\":1\" python src/video_mocap/evaluation/qualitative.py <arguments>
"""

import argparse
import os

import torch
from tqdm import tqdm

from video_mocap.vis.visualize_smpl import visualize_smpl


BODY_MODEL_PATH = "./body_models/"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
    parser.add_argument("--extension", type=str, default=".gif")
    parser.add_argument("--gpu", type=int, help="GPU device", default=0)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--methods", nargs="+", required=["moshpp", "hmr", "hmr_rr", "soma", "video_mocap*"])
    parser.add_argument("--part", type=str, default=None)
    parser.add_argument("--sequences", nargs="+", type=str, help="sequence names", default=None)
    parser.add_argument("--subjects", nargs="+", type=str, help="subject names", default=None)
    parser.add_argument("--synthetic", type=str, default=None)
    parser.add_argument("--video_res", nargs=2, type=int, help="video resolution", default=(640, 480))
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.gpu))

    camera = {
        "umpm": "l",
        "cmu_kitchen_pilot": "7151062",
        "cmu_kitchen_pilot_rb": "7151062",
        "moyo_train": None,
        "moyo_val": None,
        "bmlmovi_train": None,
        "bmlmovi_val": None,
    }[args.dataset]

    mocap_freq = 30

    dataset_dir = os.path.join(args.input_dir, args.dataset)
    mocap_dir = os.path.join(dataset_dir, "mocap")
    if args.part is not None:
        mocap_dir = os.path.join(dataset_dir, "mocap_parts___" + args.part)
    elif args.synthetic is not None:
        mocap_dir = os.path.join(dataset_dir, "mocap_synthetic___" + args.synthetic)
    videos_dir = os.path.join(dataset_dir, "videos")

    smpl_dir = os.path.join(args.input_dir, args.dataset, "smpl")
    smpl_full_dir = os.path.join(args.input_dir, args.dataset + "_full", "smpl")
    hmr_dir = os.path.join(args.input_dir, args.dataset, "comparisons", "4d_humans")
    soma_dir = os.path.join(args.input_dir, args.dataset, "comparisons", "soma", "smpl")
    video_mocap_methods = sorted([x for x in args.methods if x.startswith("video_mocap")])
    if "video_mocap" in args.methods:
        video_mocap_dir = os.path.join(args.input_dir, args.dataset, "results", "video_mocap")
    elif len(video_mocap_methods) > 0:
        video_mocap_dir = os.path.join(args.input_dir, args.dataset, "results", video_mocap_methods[0])

    # file all of the files
    files = []
    subjects = os.listdir(video_mocap_dir)
    if args.subjects is not None:
        subjects = args.subjects

    for subject in subjects:
        if args.part is not None:
            sequences = os.listdir(os.path.join(video_mocap_dir, subject, args.part))
        elif args.synthetic is not None:
            sequences = os.listdir(os.path.join(video_mocap_dir, subject, "synthetic_" + args.synthetic))
        else:
            sequences = os.listdir(os.path.join(video_mocap_dir, subject))

        if args.sequences is not None:
            sequences = args.sequences

        for sequence in sequences:
            if sequence.endswith("_stageii.npz"):
                if not os.path.exists(os.path.join(smpl_dir, subject, sequence)):
                    continue

                files.append((subject, sequence.removesuffix("_stageii.npz")))

    with torch.no_grad():
        # compute MoSh++ metrics (smpl)
        if "moshpp" in args.methods:
            print("Rendering MoSh++ results")
            progress = tqdm(total=len(files))
            for subject, sequence in files:
                sequence_path = os.path.join(smpl_dir, subject, sequence + "_stageii.npz")
                #if args.part is not None:
                    #sequence_path = os.path.join(smpl_dir, subject, args.part, sequence + "_stageii.npz")
                c3d_path = os.path.join(mocap_dir, subject, sequence + ".c3d")

                if not os.path.exists(sequence_path):
                    print("Skipping MoSh++ path")
                    continue

                output_file_dir = os.path.join("results/qual/", "moshpp", subject)
                if args.part is not None:
                    output_file_dir = os.path.join("results/qual/", "moshpp", subject, args.part)
                os.makedirs(output_file_dir, exist_ok=True)
                visualize_smpl(
                    filenames=[sequence_path, c3d_path],
                    video_path=os.path.join(output_file_dir, sequence + args.extension),
                    video_res=args.video_res,
                    angle=args.angle,
                )
                progress.update(1)

        for pose_method in ["vposer", "humor", "vposer_vid", "humor_vid"]:
            # compute VPoser and HuMoR metrics
            pose_dir = os.path.join(args.input_dir, args.dataset, "comparisons", pose_method)

            if pose_method in args.methods:
                print("Evaluating", pose_method)
                metrics_map = {}
                subjects, sequences = [], []
                progress = tqdm(total=len(files))
                for subject, sequence in files:
                    sequence_path = os.path.join(pose_dir, subject, sequence + "_stageii.npz")
                    if args.part is not None:
                        sequence_path = os.path.join(pose_dir, subject, args.part, sequence + "_stageii.npz")
                    c3d_path = os.path.join(mocap_dir, subject, sequence + ".c3d")

                    if not os.path.exists(sequence_path):
                        print("Skipping " + pose_method + " path")
                        continue

                    output_file_dir = os.path.join("results/qual/", pose_method, subject)
                    if args.part is not None:
                        output_file_dir = os.path.join("results/qual/", pose_method, subject, args.part)
                    os.makedirs(output_file_dir, exist_ok=True)
                    visualize_smpl(
                        filenames=[sequence_path, c3d_path],
                        video_path=os.path.join(output_file_dir, sequence + args.extension),
                        video_res=args.video_res,
                        angle=args.angle,
                    )
                    progress.update(1)

        # compute HMR 2.0 metrics
        if "hmr" in args.methods:
            print("Rendering HMR results")
            progress = tqdm(total=len(files))
            for subject, sequence in files:
                sequence_path = os.path.join(hmr_dir, subject, sequence + "_stageii.npz")
                if args.part is not None:
                    sequence_path = os.path.join(hmr_dir, subject, args.part, sequence + "_stageii.npz")
                c3d_path = os.path.join(mocap_dir, subject, sequence + ".c3d")

                if not os.path.exists(sequence_path):
                    print("Skipping HMR path")
                    continue

                output_file_dir = os.path.join("results/qual/", "hmr", subject)
                if args.part is not None:
                    output_file_dir = os.path.join("results/qual/", "hmr", subject, args.part)
                os.makedirs(output_file_dir, exist_ok=True)
                visualize_smpl(
                    filenames=[sequence_path, c3d_path],
                    video_path=os.path.join(output_file_dir, sequence + args.extension),
                    video_res=args.video_res,
                    angle=args.angle,
                )
                progress.update(1)

            progress.close()

        # compute HMR rigid registration metrics
        if "hmr_rr" in args.methods:
            print("Evaluating HMR 2.0 rigid registration")
            progress = tqdm(total=len(files))
            for subject, sequence in files:
                sequence_path = os.path.join(dataset_dir, "results", "hmr", subject, sequence + "_stageii.npz")
                if args.part is not None:
                    sequence_path = os.path.join(dataset_dir, "results", "hmr", subject, args.part, sequence + "_stageii.npz")
                c3d_path = os.path.join(mocap_dir, subject, sequence + ".c3d")

                if not os.path.exists(sequence_path):
                    print("Skipping hmr_rr path")
                    continue

                output_file_dir = os.path.join("results/qual/", "hmr_rr", subject)
                if args.part is not None:
                    output_file_dir = os.path.join("results/qual/", "hmr_rr", subject, args.part)
                os.makedirs(output_file_dir, exist_ok=True)
                visualize_smpl(
                    filenames=[sequence_path, c3d_path],
                    video_path=os.path.join(output_file_dir, sequence + args.extension),
                    video_res=args.video_res,
                    angle=args.angle,
                )
                progress.update(1)

            progress.close()

        # compute SOMA metrics
        if "soma" in args.methods:
            print("Rendering SOMA results")
            progress = tqdm(total=len(files))
            for subject, sequence in files:
                sequence_path = os.path.join(soma_dir, subject, sequence + "_stageii.npz")
                if args.part is not None:
                    sequence_path = os.path.join(soma_dir, subject, args.part, sequence + "_stageii.npz")
                c3d_path = os.path.join(mocap_dir, subject, sequence + ".c3d")

                if not os.path.exists(sequence_path):
                    print("Skipping SOMA path")
                    continue

                output_file_dir = os.path.join("results/qual/", "soma", subject)
                if args.part is not None:
                    output_file_dir = os.path.join("results/qual/", "soma", subject, args.part)
                os.makedirs(output_file_dir, exist_ok=True)
                visualize_smpl(
                    filenames=[sequence_path, c3d_path],
                    video_path=os.path.join(output_file_dir, sequence + args.extension),
                    video_res=args.video_res,
                    angle=args.angle,
                )
                progress.update(1)

            progress.close()

        # compute video mocap metrics
        for video_mocap_method in video_mocap_methods:
            print("Evaluating Video Mocap")
            video_mocap_method_dir = os.path.join(args.input_dir, args.dataset, "results", video_mocap_method)
            progress = tqdm(total=len(files))
            for subject, sequence in files:
                sequence_path = os.path.join(video_mocap_method_dir, subject, sequence + "_stageii.npz")
                if args.part is not None:
                    sequence_path = os.path.join(video_mocap_method_dir, subject, args.part, sequence + "_stageii.npz")
                elif args.synthetic is not None:
                    sequence_path = os.path.join(video_mocap_method_dir, subject, "synthetic_" + args.synthetic, sequence + "_stageii.npz")

                c3d_path = os.path.join(mocap_dir, subject, sequence + ".c3d")

                if not os.path.exists(sequence_path):
                    print("Skipping video_mocap path")
                    continue

                output_file_dir = os.path.join("results/qual/", video_mocap_method, subject)
                if args.part is not None:
                    output_file_dir = os.path.join("results/qual/", video_mocap_method, subject, args.part)
                if args.synthetic is not None:
                    output_file_dir = os.path.join("results/qual/", video_mocap_method, subject, "synthetic_" + args.synthetic)
                os.makedirs(output_file_dir, exist_ok=True)
                visualize_smpl(
                    filenames=[sequence_path, c3d_path],
                    video_path=os.path.join(output_file_dir, sequence + args.extension),
                    video_res=args.video_res,
                    angle=args.angle,
                )
                progress.update(1)

            progress.close()
