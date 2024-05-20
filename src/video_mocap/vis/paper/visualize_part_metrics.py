import argparse
import os
import yaml

import matplotlib.pyplot as plt
import seaborn as sns


def visualize_part_metrics(
    dataset,
):
    part_names = [
        "left_arm",
        "left_leg",
        "left_shoulder",
        "right_arm",
        "right_leg",
        "right_shoulder",
    ]
    sides = [
        "left",
        "left",
        "left", 
        "right",
        "right",
        "right",
    ]
    metrics_labels = {
        "m2s": {"title": r"m2s$\downarrow$", "y": "mm"},
        "mpjpe": {"title": r"MPJPE$\downarrow$", "y": "mm"},
        "mpjve": {"title": r"MPJVE$\downarrow$", "y": "mm/s"},
    }

    data = {
        "m2s": {"part_names": [], "values": [], "sides": []},
        "mpjpe": {"part_names": [], "values": [], "sides": []},
        "mpjve": {"part_names": [], "values": [], "sides": []},
    }

    for part_name in part_names:
        filename = os.path.join("./results/stats", dataset, part_name, "video_mocap.yaml")
        with open(filename, "r") as stream:
            try:
                data_part = yaml.safe_load(stream)
            except yaml.YAMLError as error:
                print(error)
        
        side = part_name.split("_")[0]

        data["m2s"]["sides"].append(side)
        data["mpjpe"]["sides"].append(side)
        data["mpjve"]["sides"].append(side)

        part_name_wo_side = part_name.replace("left_", "").replace("right_", "")

        data["m2s"]["part_names"].append(part_name_wo_side)
        data["mpjpe"]["part_names"].append(part_name_wo_side)
        data["mpjve"]["part_names"].append(part_name_wo_side)

        data["m2s"]["values"].append(data_part["m2s"]["mean"])
        data["mpjpe"]["values"].append(data_part["mpjpe"]["mean"])
        data["mpjve"]["values"].append(data_part["mpjve"]["mean"])

    fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

    metric_index = 0
    for metric in data.keys():
        ax = sns.barplot(
            x="part_names",
            y="values",
            hue="sides",
            data=data[metric],
            ax=axes[metric_index],
        )

        # remove legend from all but first plot
        if metric_index != 0:
            ax.get_legend().remove()

        # add y-axis labels
        axes[metric_index].set_title(metrics_labels[metric]["title"])
        axes[metric_index].set_ylabel(metrics_labels[metric]["y"])

        metric_index += 1

    output_dir = "results/vis/part_metrics"
    os.makedirs(output_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, dataset + ".pdf"))
    fig.savefig(os.path.join(output_dir, dataset + ".png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
    args = parser.parse_args()

    visualize_part_metrics(args.dataset)
