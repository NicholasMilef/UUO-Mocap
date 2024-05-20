import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from video_mocap.evaluation.comparisons import add_sequence_metrics, compute_metrics, load_c3d_data, save_metrics_stats_yaml, smpl_inference


BODY_MODEL_PATH = "./body_models/"
UNITS = "mm"
SCALE_FACTOR = {
    "m": 1,
    "cm": 100,
    "mm": 1000,
}[UNITS]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablations", nargs="+", required=["stage", "sdf"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gpu", type=int, help="GPU device", default=0)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--part", type=str, default=None)
    parser.add_argument("--subjects", nargs="+", type=str, help="subject names", default=None)
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.gpu))

    camera = {
        "umpm": "l",
        "cmu_kitchen_pilot": "7151062",
        "cmu_kitchen_pilot_rb": "7151062",
        "moyo_train": None,
        "moyo_val": None,
    }[args.dataset]

    mocap_freq = 30

    dataset_dir = os.path.join(args.input_dir, args.dataset)
    mocap_dir = os.path.join(dataset_dir, "mocap")
    videos_dir = os.path.join(dataset_dir, "videos")

    smpl_dir = os.path.join(args.input_dir, args.dataset, "smpl")
    smpl_full_dir = os.path.join(args.input_dir, args.dataset + "_full", "smpl")
    hmr_dir = os.path.join(args.input_dir, args.dataset, "comparisons", "4d_humans")
    soma_dir = os.path.join(args.input_dir, args.dataset, "comparisons", "soma", "smpl")
    video_mocap_dir = os.path.join(args.input_dir, args.dataset, "results", "video_mocap")

    # find all of the files
    files = []
    subjects = os.listdir(video_mocap_dir)
    if args.subjects is not None:
        subjects = [x for x in subjects if x in args.subjects]
    for subject in subjects:
        sequences = os.listdir(os.path.join(video_mocap_dir, subject))
        if args.part is not None:
            sequences = os.listdir(os.path.join(video_mocap_dir, subject, args.part))

        for sequence in sequences:
            if sequence.endswith("_stageii.npz"):
                if not os.path.exists(os.path.join(smpl_dir, subject, sequence)):
                    continue

                files.append((subject, sequence.removesuffix("_stageii.npz")))

    output_dir = os.path.join("results/stats", args.dataset)
    if args.part is not None:
        output_dir = os.path.join("results/stats", args.dataset, args.part)

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        # compute stage ablations
        if "stage" in args.ablations:
            print("Evaluating stage ablations")
            for stage in ["part", "chamfer", "marker", "marker_final"]:
                metrics_map = {}
                print("Evaluating stage:", stage)
                progress = tqdm(total=len(files))
                for subject, sequence in files:
                    stages = []
                    video_mocap_stage_data = np.load(os.path.join(video_mocap_dir, subject, sequence + "_stageii." + stage + ".npz"))
                    gt_data = np.load(os.path.join(smpl_dir, subject, sequence + "_stageii.npz"))
                    marker_data = load_c3d_data(os.path.join(mocap_dir, subject, sequence + ".c3d"))
                    num_frames = marker_data.shape[0]

                    pred_output = smpl_inference(video_mocap_stage_data, device)
                    gt_output = smpl_inference(gt_data, device)

                    sequence_metrics = compute_metrics(
                        marker_data=marker_data,
                        pred_smpl_output=pred_output,
                        gt_smpl_output=gt_output,
                        freq=mocap_freq,
                        part=args.part,
                    )
                    add_sequence_metrics(metrics_map, sequence_metrics)
                    progress.update(1)

                progress.close()
                save_metrics_stats_yaml(output_dir, "video_mocap_stage_" + stage + ".yaml", metrics_map)

        # compute stage ablations
        if "sdf" in args.ablations:
            print("Evaluating stage ablations")
            for method_name in ["video_mocap", "video_mocap_wo_sdf"]:
                metrics_map = {}
                print("Evaluating SDF method:", method_name)
                progress = tqdm(total=len(files))
                for subject, sequence in files:
                    video_mocap_dir_temp = os.path.join(args.input_dir, args.dataset, "results", method_name)

                    if not os.path.exists(os.path.join(video_mocap_dir_temp, subject, sequence + "_stageii.npz")):
                        #print("Skipping", os.path.join(video_mocap_dir_temp, subject, sequence + "_stageii.npz"))
                        progress.update(1)
                        continue

                    video_mocap_stage_data = np.load(os.path.join(video_mocap_dir_temp, subject, sequence + "_stageii.npz"))
                    gt_data = np.load(os.path.join(smpl_dir, subject, sequence + "_stageii.npz"))
                    marker_data = load_c3d_data(os.path.join(mocap_dir, subject, sequence + ".c3d"))
                    num_frames = marker_data.shape[0]

                    pred_output = smpl_inference(video_mocap_stage_data, device)
                    gt_output = smpl_inference(gt_data, device)

                    sequence_metrics = compute_metrics(
                        marker_data=marker_data,
                        pred_smpl_output=pred_output,
                        gt_smpl_output=gt_output,
                        freq=mocap_freq,
                        part=args.part,
                    )
                    add_sequence_metrics(metrics_map, sequence_metrics)
                    progress.update(1)

                progress.close()
                save_metrics_stats_yaml(output_dir, method_name + ".yaml", metrics_map)
