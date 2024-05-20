# Datasets
There are three main datasets used in this paper. Due to licensing, UMPM and MOYO cannot be redistributed. However, we release the code to preprocess these datasets. See the data folder in the repository for how the directory should be setup.

## HMR 2.0
Our method relies on the result of [HMR 2.0](https://github.com/shubham-goel/4D-Humans). Use this code to generate the .pkl file that extracts SMPL parameters.

## CMU Kitchen Pilot
Download the [preprocessed dataset](https://drive.google.com/drive/folders/1S-pHcope9VYS-h9ye2LlV2jpfYYEXELI?usp=drive_link)

### [Optional] Preprocessing Commands
To regenerate this data, run the following command to extract full-body mocap:
```
python src/video_mocap/datasets/preprocess_cmu_kitchen.py --input_dir <src_folder> --output_dir <data_folder> --window 15 --padding 5 5 --remove_backpack
```

## UMPM
Run the following command to extract full-body mocap:
```
python src/video_mocap/datasets/preprocess_umpm.py --input_dir <src_folder> --output_dir <data_folder> --window 15 --padding 5 5
```

Run the following command to extract partial-body mocap:
```
python src/video_mocap/datasets/preprocess_umpm.py --input_dir <src_folder> --output_dir <data_folder> --window 15 --padding 5 5 --parts --skip_video
```

## MOYO
Run the following command to extract full-body mocap:
```
python src/video_mocap/datasets/preprocess_moyo.py --input_dir <src_folder> --output_dir <data_folder> --window 3 --padding 2 2 --freq 30 --split val
```

## Ground Truth
To compute the ground-truth data, follow the commands in [SOMA](https://github.com/nghorbani/soma). Run just MoSh++ (not SOMA because SOMA estimates the markers).

MoSh++ exports to SMPL-X. However, our code and other code (HMR 2.0, HuMoR, VPoser) use SMPL-H. To compare properly, we convert from SMPL-X to SMPL using the [conversion tools](https://github.com/vchoutas/smplx/blob/main/transfer_model/docs/transfer.md). Similarly, this conversion needs to be applied to SOMA's output if benchmarking against SOMA.