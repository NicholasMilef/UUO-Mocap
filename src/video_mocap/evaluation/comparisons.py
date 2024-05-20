import argparse
import contextlib
import csv
import joblib
import yaml
import os

import ezc3d
import numpy as np
import smplx
import torch
from tqdm import tqdm

from video_mocap.datasets.dataset_utils import get_camera_name
import video_mocap.evaluation.metrics as metrics
from video_mocap.img_smpl.img_smpl import ImgSmpl
from video_mocap.utils.smpl_utils import get_joint_id, get_joint_name


BODY_MODEL_PATH = "./body_models/"
UNITS = "mm"
SCALE_FACTOR = {
    "m": 1,
    "cm": 100,
    "mm": 1000,
}[UNITS]

parts_map = {
    "left_arm": ["left_shoulder", "left_elbow", "left_wrist"],
    "left_forearm": ["left_elbow", "left_wrist"],
    "left_leg": ["left_hip", "left_knee", "left_ankle", "left_foot"],
    "left_shoulder": ["spine3", "left_collar", "left_shoulder", "left_elbow"],
    "head": ["right_shoulder", "right_elbow", "right_wrist"],
    "right_arm": ["right_shoulder", "right_elbow", "right_wrist"],
    "right_forearm": ["right_elbow", "right_wrist"],
    "right_leg": ["right_hip", "right_knee", "right_ankle", "right_foot"],
    "right_shoulder": ["spine3", "right_collar", "right_shoulder", "right_elbow"],
}

for part_name, joint_names in parts_map.items():
    ids = []
    for joint_name in joint_names:
        ids.append(get_joint_id(joint_name))
    parts_map[part_name] = ids


def compute_metrics(
    marker_data,
    pred_smpl_output,
    gt_smpl_output,
    freq,
    part,
):
    if part is not None:
        return compute_metrics_part(marker_data, pred_smpl_output, gt_smpl_output, freq, part)
    else:
        return compute_metrics_full(marker_data, pred_smpl_output, gt_smpl_output, freq)


def compute_metrics_part(
    marker_data,
    pred_smpl_output,
    gt_smpl_output,
    freq,
    part,
):
    joints = parts_map[part]
    
    output = {}
    
    num_frames = pred_smpl_output["vertices"].shape[0]
    device = pred_smpl_output["vertices"].device

    # compute m2s
    distance = metrics.compute_marker_to_surface_distance(
        pred_smpl_output["vertices"],
        torch.repeat_interleave(torch.unsqueeze(pred_smpl_output["faces"], 0), dim=0, repeats=num_frames),
        torch.from_numpy(marker_data).float().to(device),
    )
    output["m2s"] = distance * SCALE_FACTOR

    # compute MPJPE
    mpjpe = metrics.compute_MPJPE_joints(
        pred_smpl_output["joints"][:, :22],
        gt_smpl_output["joints"][:, :22],
        joints,
    )
    output["mpjpe"] = mpjpe * SCALE_FACTOR

    # compute MPJVE
    mpjve = metrics.compute_MPJVE_joints(
        pred_smpl_output["joints"][:, :22],
        gt_smpl_output["joints"][:, :22],
        freq,
        joints,
    )
    output["mpjve"] = mpjve * SCALE_FACTOR

    return output


def compute_metrics_full(
    marker_data,
    pred_smpl_output,
    gt_smpl_output,
    freq,
):
    output = {}
    
    num_frames = pred_smpl_output["vertices"].shape[0]
    device = pred_smpl_output["vertices"].device

    # compute m2s
    distance = metrics.compute_marker_to_surface_distance(
        pred_smpl_output["vertices"],
        torch.repeat_interleave(torch.unsqueeze(pred_smpl_output["faces"], 0), dim=0, repeats=num_frames),
        torch.from_numpy(marker_data).float().to(device),
    )
    output["m2s"] = distance * SCALE_FACTOR

    # compute MPJPE
    mpjpe = metrics.compute_MPJPE(
        pred_smpl_output["joints"][:, :22],
        gt_smpl_output["joints"][:, :22],
    )
    output["mpjpe"] = mpjpe * SCALE_FACTOR

    # compute PA-MPJPE
    pa_mpjpe = metrics.compute_PA_MPJPE(
        pred_smpl_output["joints"][:, :22],
        gt_smpl_output["joints"][:, :22],
    )
    output["pa_mpjpe"] = pa_mpjpe * SCALE_FACTOR

    # compute MPJVE
    mpjve = metrics.compute_MPJVE(
        pred_smpl_output["joints"][:, :22],
        gt_smpl_output["joints"][:, :22],
        freq,
    )
    output["mpjve"] = mpjve * SCALE_FACTOR

    # compute PA-MPJVE
    pa_mpjve = metrics.compute_PA_MPJVE(
        pred_smpl_output["joints"][:, :22],
        gt_smpl_output["joints"][:, :22],
        freq,
    )
    output["pa_mpjve"] = pa_mpjve * SCALE_FACTOR

    # compute V2V
    v2v = metrics.compute_V2V(
        pred_smpl_output["vertices"],
        gt_smpl_output["vertices"],
    )
    output["v2v"] = v2v * SCALE_FACTOR

    return output


def load_c3d_data(filename):
    marker_data = ezc3d.c3d(filename)

    scale_factor = {
        "m": 1,
        "cm": 100,
        "mm": 1000,
    }[marker_data["parameters"]["POINT"]["UNITS"]["value"][0]]

    points = marker_data["data"]["points"] / scale_factor
    points = points[:3, :, :].transpose((2, 1, 0))
    return points


def smplx_inference(smplx_data, device):
    num_frames = smplx_data["trans"].shape[0]

    smplx_model = smplx.create(
        BODY_MODEL_PATH,
        model_type = "smplx",
        num_betas=10,
        gender = smplx_data["gender"].item(),
        batch_size = num_frames,
    ).to(device)

    output = {}

    body_pose = smplx_data["poses"][:, 3:66]
    betas = np.repeat(np.expand_dims(smplx_data["betas"][:10], 0), axis=0, repeats=num_frames)
    global_orient = smplx_data["poses"][:, :3]
    transl = smplx_data["trans"]

    smplx_output = smplx_model(
        body_pose = torch.from_numpy(body_pose).to(device).float(),
        betas = torch.from_numpy(betas).to(device).float(),
        global_orient = torch.from_numpy(global_orient).to(device).float(),
        transl = torch.from_numpy(transl).to(device).float(),
    )
    output["joints"] = smplx_output["joints"]
    output["vertices"] = smplx_output["vertices"]
    output["faces"] = torch.from_numpy(smplx_model.faces.astype(np.int32)).to(device)

    return output


def smpl_inference(
    smpl_data,
    device,
    zero_hand=True,
):
    num_frames = smpl_data["trans"].shape[0]

    # prevent warning messages from showing
    with contextlib.redirect_stdout(None):
        smpl_model = smplx.create(
            BODY_MODEL_PATH,
            model_type = "smpl",
            num_betas=10,
            gender = smpl_data["gender"].item(),
            batch_size = num_frames,
        ).to(device)

    output = {}

    poses = smpl_data["poses"]
    if len(smpl_data["poses"]) == 3:
        poses = np.reshape(smpl_data["poses"], (smpl_data["poses"].shape[0], -1))

    body_pose = poses[:, 3:72]
    betas = np.repeat(np.expand_dims(smpl_data["betas"][:10], 0), axis=0, repeats=num_frames)
    global_orient = poses[:, :3]
    transl = smpl_data["trans"]

    # to ensure fairness across multiple methods, we zero out the hands
    # (HuMoR and VPoser produce 21 joints rather than 23)
    if zero_hand:
        body_pose[:, 63:69] = 0

    smpl_output = smpl_model(
        body_pose = torch.from_numpy(body_pose).to(device).float(),
        betas = torch.from_numpy(betas).to(device).float(),
        global_orient = torch.from_numpy(global_orient).to(device).float(),
        transl = torch.from_numpy(transl).to(device).float(),
    )
    output["joints"] = smpl_output["joints"]
    output["vertices"] = smpl_output["vertices"]
    output["faces"] = torch.from_numpy(smpl_model.faces.astype(np.int32)).to(device)

    return output


def add_sequence_metrics(metrics_map, sequence_metrics):
    for key, value in sequence_metrics.items():
        if key not in metrics_map:
            metrics_map[key] = []
        metrics_map[key].append(value)


def save_metrics_stats_yaml(output_dir, filename, metrics_map):
    output = {}
    for key in metrics_map.keys():
        values = torch.stack(metrics_map[key], dim=0)
        if len(values.shape) == 1:
            output[key] = {
                "mean": round(float(torch.mean(values).item()), 1),
                "std": round(float(torch.std(values).item()), 1),
                "median": round(float(torch.median(values).item()), 1),
            }
        else:
            output[key] = {}
            for i in range(values.shape[-1]):
                output[key][get_joint_name(i)] = {
                    "mean": round(float(torch.mean(values, dim=0)[i].item()), 1),
                    "std": round(float(torch.std(values, dim=0)[i].item()), 1),
                    "median": round(float(torch.median(values, dim=0)[0][i].item()), 1),
                }

    with open(os.path.join(output_dir, filename), "w") as f:
        yaml.dump(output, f)


def save_metrics_stats_csv(output_dir, filename, metrics_map, subjects, sequences):
    output = []
    output.append(["subject", "sequence"] + list(metrics_map.keys()))
    num_rows = len(metrics_map[list(metrics_map.keys())[0]])

    for i in range(num_rows):
        output.append([])
        output[-1].append(subjects[i])
        output[-1].append(sequences[i])
        for key in metrics_map.keys():
            value = metrics_map[key][i]
            if len(value.shape) == 1:
                output[-1].append(-1)
            else:
                output[-1].append(float(value.item()))

    with open(os.path.join(output_dir, filename), "w") as f:
        writer = csv.writer(f)
        writer.writerows(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gpu", type=int, help="GPU device", default=0)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--methods", nargs="+", required=["moshpp", "hmr", "hmr_rr", "soma", "video_mocap", "vposer", "humor", "vposer_vid", "humor_vid"])
    parser.add_argument("--part", type=str, default=None)
    parser.add_argument("--subjects", nargs="+", type=str, help="subject names", default=None)
    parser.add_argument("--synthetic", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.gpu))

    camera = get_camera_name(args.dataset)

    mocap_freq = 30

    dataset_dir = os.path.join(args.input_dir, args.dataset)
    if args.part is not None:
        mocap_dir = os.path.join(dataset_dir, "mocap_parts___" + args.part)
    elif args.synthetic is not None:
        mocap_dir = os.path.join(dataset_dir, "mocap_synthetic___" + args.synthetic)
    else:
        mocap_dir = os.path.join(dataset_dir, "mocap")

    videos_dir = os.path.join(dataset_dir, "videos")

    smpl_dir = os.path.join(args.input_dir, args.dataset, "smpl")
    smpl_full_dir = os.path.join(args.input_dir, args.dataset + "_full", "smpl")
    hmr_dir = os.path.join(args.input_dir, args.dataset, "comparisons", "4d_humans")
    soma_dir = os.path.join(args.input_dir, args.dataset, "comparisons", "soma", "smpl")
    video_mocap_methods = [x for x in args.methods if x.startswith("video_mocap")]
    if "video_mocap" in args.methods:
        video_mocap_dir = os.path.join(args.input_dir, args.dataset, "results", "video_mocap")
    elif len(video_mocap_methods) > 0:
        video_mocap_dir = os.path.join(args.input_dir, args.dataset, "results", video_mocap_methods[0])

    # find all of the files
    files = []
    subjects = os.listdir(video_mocap_dir)
    if args.subjects is not None:
        subjects = [x for x in subjects if x in args.subjects]
    for subject in subjects:
        if args.part is not None:
            sequences = os.listdir(os.path.join(video_mocap_dir, subject, args.part))
        elif args.synthetic is not None:
            sequences = os.listdir(os.path.join(video_mocap_dir, subject, "synthetic_" + args.synthetic))
        else:
            sequences = os.listdir(os.path.join(video_mocap_dir, subject))

        for sequence in sequences:
            if sequence.endswith("_stageii.npz"):
                if not os.path.exists(os.path.join(smpl_dir, subject, sequence)):
                    continue

                files.append((subject, sequence.removesuffix("_stageii.npz")))

    if args.part is not None:
        output_dir = os.path.join("results/stats", args.dataset, args.part)
    elif args.synthetic is not None:
        output_dir = os.path.join("results/stats", args.dataset, "synthetic_" + args.synthetic)
    else:
        output_dir = os.path.join("results/stats", args.dataset)

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        # compute MoSh++ metrics (smpl)
        if "moshpp" in args.methods:
            print("Evaluating MoSh++ (Reference)")
            metrics_map = {}
            subjects, sequences = [], []
            progress = tqdm(total=len(files))
            for subject, sequence in files:
                smpl_data = np.load(os.path.join(smpl_dir, subject, sequence + "_stageii.npz"))
                marker_data = load_c3d_data(os.path.join(mocap_dir, subject, sequence + ".c3d"))
                num_frames = marker_data.shape[0]

                smpl_output = smpl_inference(smpl_data, device)

                sequence_metrics = compute_metrics(
                    marker_data=marker_data,
                    pred_smpl_output=smpl_output,
                    gt_smpl_output=smpl_output,
                    freq=mocap_freq,
                    part=args.part,
                )
                add_sequence_metrics(metrics_map, sequence_metrics)
                subjects.append(subject)
                sequences.append(sequence)
                progress.update(1)

            progress.close()
            save_metrics_stats_yaml(output_dir, "moshpp.yaml", metrics_map)
            save_metrics_stats_csv(output_dir, "moshpp.csv", metrics_map, subjects, sequences)

        for pose_method in ["vposer", "humor", "vposer_vid", "humor_vid"]:
            # compute VPoser and HuMoR metrics
            pose_dir = os.path.join(args.input_dir, args.dataset, "comparisons", pose_method)

            if pose_method in args.methods:
                print("Evaluating", pose_method)
                metrics_map = {}
                subjects, sequences = [], []
                progress = tqdm(total=len(files))
                for subject, sequence in files:
                    gt_data = np.load(os.path.join(smpl_dir, subject, sequence + "_stageii.npz"))
                    marker_data = load_c3d_data(os.path.join(mocap_dir, subject, sequence + ".c3d"))
                    smpl_data = np.load(os.path.join(pose_dir, subject, sequence + "_stageii.npz"))
                    num_frames = marker_data.shape[0]

                    pred_output = smpl_inference(smpl_data, device)
                    gt_output = smpl_inference(gt_data, device)

                    sequence_metrics = compute_metrics(
                        marker_data=marker_data,
                        pred_smpl_output=pred_output,
                        gt_smpl_output=gt_output,
                        freq=mocap_freq,
                        part=args.part,
                    )
                    add_sequence_metrics(metrics_map, sequence_metrics)
                    subjects.append(subject)
                    sequences.append(sequence)
                    progress.update(1)

                progress.close()
                save_metrics_stats_yaml(output_dir, pose_method + ".yaml", metrics_map)
                save_metrics_stats_csv(output_dir, pose_method + ".csv", metrics_map, subjects, sequences)

        # compute HMR 2.0 metrics
        if "hmr" in args.methods:
            print("Evaluating HMR 2.0")
            metrics_map = {}
            subjects, sequences = [], []
            progress = tqdm(total=len(files))
            for subject, sequence in files:
                gt_data = np.load(os.path.join(smpl_dir, subject, sequence + "_stageii.npz"))
                marker_data = load_c3d_data(os.path.join(mocap_dir, subject, sequence + ".c3d"))
                hmr_data = joblib.load(os.path.join(hmr_dir, subject, sequence + "." + camera, "results", "demo_" + sequence + ".pkl"))
                hmr_data = ImgSmpl(hmr_data, mocap_freq).get_smpl()
                num_frames = marker_data.shape[0]

                pred_output = smpl_inference(hmr_data, device)
                gt_output = smpl_inference(gt_data, device)

                sequence_metrics = compute_metrics(
                    marker_data=marker_data,
                    pred_smpl_output=pred_output,
                    gt_smpl_output=gt_output,
                    freq=mocap_freq,
                    part=args.part,
                )
                add_sequence_metrics(metrics_map, sequence_metrics)
                subjects.append(subject)
                sequences.append(sequence)
                progress.update(1)

            progress.close()
            save_metrics_stats_yaml(output_dir, "hmr.yaml", metrics_map)
            save_metrics_stats_csv(output_dir, "hmr.csv", metrics_map, subjects, sequences)

        # compute HMR rigid registration metrics
        if "hmr_rr" in args.methods:
            print("Evaluating HMR 2.0 rigid registration")
            metrics_map = {}
            subjects, sequences = [], []
            progress = tqdm(total=len(files))
            for subject, sequence in files:
                if args.part is not None:
                    sequence_path = os.path.join(dataset_dir, "results", "hmr", subject, args.part, sequence + "_stageii.npz")
                elif args.synthetic is not None:
                    sequence_path = os.path.join(dataset_dir, "results", "hmr", subject, "synthetic_" + args.synthetic, sequence + "_stageii.npz")
                else:
                    sequence_path = os.path.join(dataset_dir, "results", "hmr", subject, sequence + "_stageii.npz")

                gt_data = np.load(os.path.join(smpl_dir, subject, sequence + "_stageii.npz"))
                marker_data = load_c3d_data(os.path.join(mocap_dir, subject, sequence + ".c3d"))
                hmr_data = np.load(sequence_path)
                num_frames = marker_data.shape[0]

                pred_output = smpl_inference(hmr_data, device)
                gt_output = smpl_inference(gt_data, device)

                sequence_metrics = compute_metrics(
                    marker_data=marker_data,
                    pred_smpl_output=pred_output,
                    gt_smpl_output=gt_output,
                    freq=mocap_freq,
                    part=args.part,
                )
                add_sequence_metrics(metrics_map, sequence_metrics)
                subjects.append(subject)
                sequences.append(sequence)
                progress.update(1)

            progress.close()
            save_metrics_stats_yaml(output_dir, "hmr_rr.yaml", metrics_map)
            save_metrics_stats_csv(output_dir, "hmr_rr.csv", metrics_map, subjects, sequences)

        # compute SOMA metrics
        if "soma" in args.methods:
            print("Evaluating SOMA")
            metrics_map = {}
            subjects, sequences = [], []
            progress = tqdm(total=len(files))
            for subject, sequence in files:
                if args.part is not None:
                    sequence_path = os.path.join(soma_dir, subject, args.part, sequence + "_stageii.npz")
                elif args.synthetic is not None:
                    sequence_path = os.path.join(soma_dir, subject, "synthetic_" + args.synthetic, sequence + "_stageii.npz")
                else:
                    sequence_path = os.path.join(soma_dir, subject, sequence + "_stageii.npz")

                if not os.path.exists(sequence_path):
                    print("Skipping SOMA path")
                    continue

                gt_data = np.load(os.path.join(smpl_dir, subject, sequence + "_stageii.npz"))
                marker_data = load_c3d_data(os.path.join(mocap_dir, subject, sequence + ".c3d"))
                soma_data = np.load(sequence_path)
                num_frames = marker_data.shape[0]

                pred_output = smpl_inference(soma_data, device)
                gt_output = smpl_inference(gt_data, device)

                sequence_metrics = compute_metrics(
                    marker_data=marker_data,
                    pred_smpl_output=pred_output,
                    gt_smpl_output=gt_output,
                    freq=mocap_freq,
                    part=args.part,
                )
                add_sequence_metrics(metrics_map, sequence_metrics)
                subjects.append(subject)
                sequences.append(sequence)
                progress.update(1)

            progress.close()
            save_metrics_stats_yaml(output_dir, "soma.yaml", metrics_map)
            save_metrics_stats_csv(output_dir, "soma.csv", metrics_map, subjects, sequences)

        # compute video mocap metrics
        video_mocap_methods = [x for x in args.methods if x.startswith("video_mocap")]
        for video_mocap_method in video_mocap_methods:
            print("Evaluating Video Mocap:", video_mocap_method)
            video_mocap_method_dir = os.path.join(args.input_dir, args.dataset, "results", video_mocap_method)
            metrics_map = {}
            subjects, sequences = [], []
            progress = tqdm(total=len(files))
            for subject, sequence in files:
                if args.part is not None:
                    sequence_path = os.path.join(video_mocap_method_dir, subject, args.part, sequence + "_stageii.npz")
                elif args.synthetic is not None:
                    sequence_path = os.path.join(video_mocap_method_dir, subject, "synthetic_" + args.synthetic, sequence + "_stageii.npz")
                else:
                    sequence_path = os.path.join(video_mocap_method_dir, subject, sequence + "_stageii.npz")

                if not os.path.exists(sequence_path):
                    print("Skipping " + video_mocap_method + " path")
                    continue

                gt_data = np.load(os.path.join(smpl_dir, subject, sequence + "_stageii.npz"))
                marker_data = load_c3d_data(os.path.join(mocap_dir, subject, sequence + ".c3d"))
                video_mocap_data = np.load(sequence_path)
                num_frames = marker_data.shape[0]

                pred_output = smpl_inference(video_mocap_data, device)
                gt_output = smpl_inference(gt_data, device)

                sequence_metrics = compute_metrics(
                    marker_data=marker_data,
                    pred_smpl_output=pred_output,
                    gt_smpl_output=gt_output,
                    freq=mocap_freq,
                    part=args.part,
                )
                add_sequence_metrics(metrics_map, sequence_metrics)
                subjects.append(subject)
                sequences.append(sequence)
                progress.update(1)

            progress.close()
            save_metrics_stats_yaml(output_dir, video_mocap_method + ".yaml", metrics_map)
            save_metrics_stats_csv(output_dir, video_mocap_method + ".csv", metrics_map, subjects, sequences)
