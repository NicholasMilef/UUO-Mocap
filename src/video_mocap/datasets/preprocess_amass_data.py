import argparse
import os

import torch
from tqdm import tqdm
import numpy as np

from video_mocap.utils.foot_contact import compute_foot_contacts_np
from video_mocap.utils.smpl import SmplInferenceGender


def resample_sequence(data, src_freq, tgt_freq):
    output_data = {}
    
    stride = round(float(src_freq)) // int(tgt_freq)
    for key, _ in data.items():
        if key in ["trans", "poses", "dmpls"]:
            output_data[key] = np.ascontiguousarray(data[key][::stride])
        else:
            output_data[key] = data[key]

    output_data["mocap_framerate"] = tgt_freq
    return output_data


def convert_to_type(data, src_dtype, tgt_dtype):
    output_data = {}

    for key, _ in data.items():
        if isinstance(data[key], np.ndarray) and data[key].dtype == src_dtype:
            output_data[key] = data[key].astype(tgt_dtype)
        else:
            output_data[key] = data[key]

    return output_data


def preprocess(args):
    device = torch.device("cpu")
    if args.gpu is not None:
        device = torch.device(args.gpu)

    root_dir = "./data/SMPL_H_G/"
    output_dir = "./data/processed/SMPL_H_G/"

    smpl_inference = SmplInferenceGender(device)

    # count number of files
    print("Counting files...")
    num_samples = 0
    for dataset_name in os.listdir(root_dir):
        if not os.path.isdir(os.path.join(root_dir, dataset_name)):
            continue
        subjects = os.listdir(os.path.join(root_dir, dataset_name))
        for subject in subjects:
            if not os.path.isdir(os.path.join(root_dir, dataset_name, subject)):
                continue
            sequences = os.listdir(os.path.join(root_dir, dataset_name, subject))
            for sequence in sequences:
                num_samples += 1

    # process each file
    num_processed_samples = 0
    print("Preprocessing data...")
    progress = tqdm(total=num_samples)
    for dataset_name in os.listdir(root_dir):
        if not os.path.isdir(os.path.join(root_dir, dataset_name)):
            continue
        subjects = os.listdir(os.path.join(root_dir, dataset_name))
        for subject in subjects:
            if not os.path.isdir(os.path.join(root_dir, dataset_name, subject)):
                continue
            sequences = os.listdir(os.path.join(root_dir, dataset_name, subject))
            for sequence in sequences:
                filename = os.path.join(root_dir, dataset_name, subject, sequence)
                sequence_data = np.load(filename)

                # skip data if no animation data
                if "poses" in sequence_data:
                    if args.tgt_freq:
                        sequence_data = resample_sequence(sequence_data, sequence_data["mocap_framerate"], args.tgt_freq)

                    sequence_data = convert_to_type(sequence_data, np.float64, np.float32)

                    num_frames = sequence_data["trans"].shape[0]
                    gender_one_hot = torch.zeros((1, 2)).to(device)

                    if sequence_data["gender"].item() == "male":
                        gender_one_hot[:, 0] = 1
                    elif sequence_data["gender"].item() == "male":
                        gender_one_hot[:, 1] = 1

                    betas = torch.from_numpy(sequence_data["betas"])
                    betas = torch.unsqueeze(betas, 0)

                    if num_frames < args.min_length:
                        progress.update(1)
                        continue

                    smpl_output = smpl_inference(
                        poses=torch.from_numpy(sequence_data["poses"][:, 3:72]).to(device).unsqueeze(0),
                        betas=betas[:, :10].to(device),
                        root_orient=torch.from_numpy(sequence_data["poses"][:, 0:3]).to(device).unsqueeze(0),
                        trans=torch.from_numpy(sequence_data["trans"]).to(device).unsqueeze(0),
                        gender_one_hot=gender_one_hot,
                    )
                    sequence_data["foot_contacts"] = compute_foot_contacts_np(
                        smpl_output["joints"].detach().cpu().numpy(),
                    ).astype(np.float32)

                    output_filename = os.path.join(output_dir, dataset_name, subject, sequence)
                    os.makedirs(os.path.join(output_dir, dataset_name, subject), exist_ok=True)
                    np.savez(output_filename, **sequence_data)

                    num_processed_samples += 1

                progress.update(1)

    progress.close()
    print("Processed", num_processed_samples, "samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--min_length", type=int, default=64)
    parser.add_argument("--tgt_freq", type=int, help="target frequency", default=30)
    args = parser.parse_args()

    preprocess(args)
