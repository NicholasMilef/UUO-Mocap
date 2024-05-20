import argparse
import os

from PIL import Image


def crop_results(args):
    if args.part is not None:
        dir_name = os.path.join(
            "paper/results_section/",
            args.dataset,
            args.subject,
            args.sequence,
            args.part,
        )
    else:
        dir_name = os.path.join(
            "paper/results_section/",
            args.dataset,
            args.subject,
            args.sequence,
        )
    os.makedirs(dir_name, exist_ok=True)

    frame_name = str(args.frame).zfill(8) + ".png"

    for method in args.methods:
        output_filename = os.path.join(dir_name, method + ".png")
    
        if args.part is not None and method != "moshpp":
            input_filename = os.path.join(
                "results/qual",
                method,
                args.subject,
                args.part,
                args.sequence,
                frame_name,
            )
        else:
            input_filename = os.path.join(
                "results/qual",
                method,
                args.subject,
                args.sequence,
                frame_name,
            )                

        image = Image.open(input_filename)
        width_crop = image.width * args.scale
        height_crop = image.height * args.scale
        center_crop = (
            image.width * args.center[0],
            image.height * args.center[1],
        )

        if args.portrait:
            image_crop = image.crop((
                center_crop[0] - (height_crop / 2),
                center_crop[1] - (width_crop / 2),
                center_crop[0] + (height_crop / 2),
                center_crop[1] + (width_crop / 2),
            ))
        else:
            image_crop = image.crop((
                center_crop[0] - (width_crop / 2),
                center_crop[1] - (height_crop / 2),
                center_crop[0] + (width_crop / 2),
                center_crop[1] + (height_crop / 2),
            ))
        image_crop.save(output_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--center", nargs=2, type=float, help="center of crop", required=True)
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--methods", nargs="+", type=str, help="dataset", required=True)
    parser.add_argument("--part", type=str, help="part name", default=None)
    parser.add_argument("--portrait", action="store_true", help="use portrait mode")
    parser.add_argument("--scale", type=float, help="scale of crop in relation to image", default=0.2)
    parser.add_argument("--sequence", type=str, help="sequence", required=True)
    parser.add_argument("--subject", type=str, help="subject", required=True)
    args = parser.parse_args()

    crop_results(args)
