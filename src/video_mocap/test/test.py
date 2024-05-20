import argparse
import joblib
import os
from typing import Dict, List

import cv2
import numpy as np
from pytorch3d.transforms import matrix_to_axis_angle
import torch

from video_mocap.datasets.preprocess_cmu_kitchen import cleanup_markers
from video_mocap.img_smpl.img_smpl import ImgSmpl
from video_mocap.markers.markers import Markers
from video_mocap.multimodal import multimodal_video_mocap
from video_mocap.utils.config import load_config


def test(
    input_dir: str,
    output_dir: str,
    dataset: str,
    camera: str,
    config: Dict,
    part: str=None,
    synthetic: str=None,
    sequences: List[str] = None,
    subjects: List[str] = None,
    num_files=None
):
    device = torch.device("cpu")
    if torch.cuda.is_available() and not args.cpu_only:
        device = torch.device("cuda:" + str(args.gpu))

    if part:
        mocap_dir = os.path.join(input_dir, dataset, "mocap_parts___" + part)
    elif synthetic:
        mocap_dir = os.path.join(input_dir, dataset, "mocap_synthetic___" + synthetic)
    else:
        mocap_dir = os.path.join(input_dir, dataset, "mocap")

    video_dir = os.path.join(input_dir, dataset, "videos")
    comparisons_dir = os.path.join(input_dir, dataset, "comparisons", "4d_humans")

    if subjects is None:
        subjects = sorted(os.listdir(mocap_dir))

    listed_sequences = sequences

    file_count = 0
    for subject in subjects:
        if listed_sequences is None:
            sequences = sorted(os.listdir(os.path.join(mocap_dir, subject)))
        else:
            sequences = [(x+".c3d") for x in listed_sequences]

        sequences = [x for x in sequences if x.endswith(".c3d")]
        for sequence in sequences:
            sequence_name = sequence.split(".c3d")[0]
            video_sequence_name = sequence_name
            if camera is not None:
                video_sequence_name = video_sequence_name + "." + camera

            # create export directory
            if part:
                output_filename = os.path.join(output_dir, subject, sequence_name + "_stageii")
            elif synthetic:
                output_filename = os.path.join(output_dir, subject, "synthetic_" + synthetic, sequence_name + "_stageii")
            else:
                output_filename = os.path.join(output_dir, subject, sequence_name + "_stageii")

            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            if os.path.exists(output_filename + ".npz"):
                print("Skipping", output_filename)
                continue

            markers_filename = os.path.join(mocap_dir, subject, sequence)
            smpl_video_filename = os.path.join(video_dir, subject, video_sequence_name + ".avi")
            smpl_img_filename = os.path.join(
                comparisons_dir,
                subject,
                video_sequence_name,
                #"outputs_" + dataset,
                "results",
                "demo_" + sequence_name + ".pkl",
            )
            # get source video data
            video = cv2.VideoCapture(smpl_video_filename)
            video_freq = video.get(cv2.CAP_PROP_FPS)

            # SMPL data from monocular
            if not os.path.isfile(smpl_img_filename):
                print("Skipping", smpl_img_filename)
                continue

            smpl_img_data = joblib.load(smpl_img_filename)
            img_smpl = ImgSmpl(smpl_img_data, video_freq)

            markers = Markers(markers_filename)
            points = markers.get_points()
            points = np.nan_to_num(points, 0)
            points = cleanup_markers(points)
            markers.set_points(points)

            multimodal_smpl = multimodal_video_mocap(
                img_smpl,
                markers,
                device,
                config=config,
                offset=0,  # datasets are aligned, so no offset needed
                print_options=args.print_options,
                save_stages=True,
            )
            
            # save output
            smpl_output = {}
            smpl_output["betas"] = multimodal_smpl["betas"][0]
            smpl_output["trans"] = multimodal_smpl["trans"]
            smpl_output["poses"] = torch.cat((multimodal_smpl["root_orient"], multimodal_smpl["pose_body"]), dim=1)
            smpl_output["poses"] = torch.flatten(matrix_to_axis_angle(smpl_output["poses"]), start_dim=1, end_dim=-1)
            smpl_output["mocap_frame_rate"] = multimodal_smpl["mocap_frame_rate"]
            smpl_output["mocap_markers"] = multimodal_smpl["mocap_markers"].get_points()
            smpl_output["gender"] = "neutral"

            # convert to numpy
            for key in smpl_output.keys():
                if isinstance(smpl_output[key], torch.Tensor):
                    smpl_output[key] = smpl_output[key].detach().cpu().numpy()
            np.savez(output_filename, **smpl_output)

            # convert to numpy
            for stage in multimodal_smpl["stages"].keys():
                stage_smpl = multimodal_smpl["stages"][stage]

                root_orient = torch.from_numpy(stage_smpl["root_orient"])
                pose_body = torch.from_numpy(stage_smpl["pose_body"])

                smpl_output["trans"] = stage_smpl["trans"]
                smpl_output["betas"] = stage_smpl["betas"]
                smpl_output["poses"] = torch.cat((root_orient, pose_body), dim=1)
                smpl_output["poses"] = torch.flatten(matrix_to_axis_angle(smpl_output["poses"]), start_dim=1, end_dim=-1).detach().cpu().numpy()
                stage_output_filename = output_filename
                stage_output_filename = stage_output_filename.replace("_stageii", "_stageii." + stage)
                np.savez(stage_output_filename, **smpl_output)

            file_count += 1
            if num_files is not None and file_count > num_files:
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="configuration file", required=True)
    parser.add_argument("--cpu_only", action="store_true", help="only use the CPU")
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
    parser.add_argument("--input_dir", type=str, help="input directory", required=True)
    parser.add_argument("--gpu", type=int, help="GPU ID")
    parser.add_argument("--num_files", type=int, help="number of files", default=None)
    parser.add_argument("--sequences", nargs="+", type=str, help="sequence names", default=None)
    parser.add_argument("--subjects", nargs="+", type=str, help="subject names", default=None)
    parser.add_argument("--synthetic", action="store_true", help="use synthetic mocap")
    parser.add_argument("--synthetic_list", nargs="+", default=[])
    parser.add_argument("--parts", action="store_true", help="use part mocap")
    parser.add_argument("--parts_list", nargs="+", default=[])
    parser.add_argument("--print_options", type=str, nargs="*", default=["loss", "progress"])
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = os.path.join(args.input_dir, args.dataset, "results", config["name"])

    camera = {
        "umpm": "l",
        "cmu_kitchen_pilot": "7151062",
        "cmu_kitchen_pilot_rb": "7151062",
        "moyo_train": None,
        "moyo_val": None,
        "bmlmovi_train": None,
        "bmlmovi_val": None,
    }[args.dataset]

    if args.parts:
        parts_dirs = os.listdir(os.path.join(args.input_dir, args.dataset))
        parts_dirs = [x for x in parts_dirs if x.startswith("mocap_parts")]
        
        if len(args.parts_list) > 0:
            trimmed_parts_dirs = []
            for parts_dir in parts_dirs:
                if parts_dir.split("mocap_parts___")[-1] in args.parts_list:
                    trimmed_parts_dirs.append(parts_dir)
            parts_dirs = trimmed_parts_dirs

        for parts_dir in parts_dirs:
            part = parts_dir.split("___")[-1]
            test(
                input_dir=args.input_dir,
                output_dir=output_dir,
                dataset=args.dataset,
                camera=camera,
                config=config,
                part=part,
                sequences=args.sequences,
                subjects=args.subjects,
                num_files=args.num_files,
            )
    elif args.synthetic:
        synthetic_dirs = os.listdir(os.path.join(args.input_dir, args.dataset))
        synthetic_dirs = [x for x in synthetic_dirs if x.startswith("mocap_synthetic")]

        if len(args.synthetic_list) > 0:
            trimmed_synthetic_dirs = []
            for synthetic_dir in synthetic_dirs:
                if synthetic_dir.split("mocap_synthetic___")[-1] in args.synthetic_list:
                    trimmed_synthetic_dirs.append(synthetic_dir)
            synthetic_dirs = trimmed_synthetic_dirs

        for synthetic_dir in synthetic_dirs:
            synthetic = synthetic_dir.split("___")[-1]
            test(
                input_dir=args.input_dir,
                output_dir=output_dir,
                dataset=args.dataset,
                camera=camera,
                config=config,
                part=None,
                synthetic=synthetic,
                sequences=args.sequences,
                subjects=args.subjects,
                num_files=args.num_files,
            )

    else:
        test(
            input_dir=args.input_dir,
            output_dir=output_dir,
            dataset=args.dataset,
            camera=camera,
            config=config,
            part=None,
            sequences=args.sequences,
            subjects=args.subjects,
            num_files=args.num_files,
        )
