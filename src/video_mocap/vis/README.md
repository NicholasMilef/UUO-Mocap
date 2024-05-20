# Visualization modules
There are many visualization modules to help with visualizing some or parts of the results. Here are basic instructions on how to run these modules. Note that many of these modules have other command-line arguments that can be further used to customize results. If a `video_path` is not supplied, then the visualization will run in interactive mode.

## Visualize SMPL sequences (with optional markers)
```python src/video_mocap/vis/visualize_smpl.py --filenames <smpl_npz_file> --video_path <output_video_filename> --marker_size 0.02```

## Visualize SMPL sequences (with optional markers)
```python src/video_mocap/vis/visualize_reprojection_loss.py --input_dir <input_dir> --dataset <dataset> --subject <subject> --sequence <sequence> --video_path <output_video_filename> --marker_size 0.02```

## Visualize model
```python src/video_mocap/vis/visualize_model.py --input_dir <input_dir> --dataset <dataset> --subject <subject> --sequence <sequence> --video_path <output_video_filename> --marker_size 0.02```