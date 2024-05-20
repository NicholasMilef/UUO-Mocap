import os
import time
from typing import Callable, List

import cv2
import imageio
import numpy as np
import pyrender
from scipy.spatial.transform import Rotation

from video_mocap.vis.quality_renderer import DirectionalLightQuality, OffscreenRendererQuality


class VideoMocapRenderer:
    def __init__(
        self,
        scene: pyrender.Scene,
        render_frame_fn: Callable,
        num_frames: int,
        data_freq: float,
        camera_matrix: np.ndarray=None,
        video_fps: int=30,
        video_res: List[int]=[640, 480],
        video_path: str=None,
        image_postprocess_fn: Callable=None,
        quality_mode: str="normal",
    ):
        """
        Renderer class that can create both interactive and offscreen renderings

        Args:
            scene: PyRender scene
            render_frame_fn: function that gets called every frame, F(frame_number)
            num_frames: number of frames in sequences (F)
            data_freq: frequency of input data
            camera_matrix: camera transformation matrix [4, 4]
            video_fps: video frames per second
            video_path: output path for video
        """

        # render mode
        render_mode = "online"
        if video_path is not None:
            render_mode = "offline"
            video_format = os.path.splitext(video_path)[1][1:]
            video_sequence = os.path.basename(os.path.splitext(video_path)[0])
            os.environ["DISPLAY"] = ":1"

        # replace directional lights with higher-quality version
        if quality_mode == "ultra":
            for node in scene.nodes:
                if node.light and isinstance(node.light, pyrender.DirectionalLight):                
                    old_light = node.light
                    node.light = DirectionalLightQuality(
                        color = old_light.color,
                        intensity = old_light.intensity,
                    )

        # setup scene
        run_in_thread = True
        if render_mode == "offline":
            run_in_thread = False

        if camera_matrix is None:
            camera_matrix = [
                [0.750, -0.252, 0.611, 3.4],
                [0.661, 0.291, -0.691, -3.0],
                [-0.004, 0.923, 0.385, 2.3],
                [0, 0, 0, 1],
            ]



        if render_mode == "online":
            viewer = pyrender.Viewer(
                scene,
                use_raymond_lighting=False,
                run_in_thread=run_in_thread,
                shadows=True,
            )

            frame = 0
            while viewer.is_active:
                viewer.render_lock.acquire()
                render_frame_fn(frame)
                viewer.render_lock.release()

                frame = (frame + 1) % num_frames
                time.sleep(1.0 / data_freq)

        elif render_mode == "offline":
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
            scene.add(camera, pose=camera_matrix)

            if quality_mode == "ultra":
                renderer = OffscreenRendererQuality(
                    viewport_width=video_res[0],
                    viewport_height=video_res[1],
                    point_size=1.0,
                )
            elif quality_mode == "normal":
                renderer = pyrender.OffscreenRenderer(
                    viewport_width=video_res[0],
                    viewport_height=video_res[1],
                    point_size=1.0,
                )

            flags = pyrender.constants.RenderFlags.SHADOWS_ALL

            os.makedirs(os.path.dirname(video_path), exist_ok=True)

            if video_format == "mp4":
                cap = cv2.VideoCapture(0)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(video_path, fourcc, video_fps, video_res)

                # write out video
                for frame in range(0, num_frames, data_freq // video_fps):
                    render_frame_fn(frame)
                    color, _ = renderer.render(scene, flags=flags)
                    color = np.flip(color, axis=2)
                    if image_postprocess_fn is not None:
                        color = image_postprocess_fn(frame, color)
                    out.write(color)

                cap.release()
                out.release()
                #cv2.destroyAllWindows()
            elif video_format == "gif":
                # write out video
                with imageio.get_writer(video_path, mode="I", duration=(1000.0 * (1.0 / video_fps))) as writer:
                    for frame in range(0, num_frames, data_freq // video_fps):
                        render_frame_fn(frame)
                        color, _ = renderer.render(scene, flags=flags)
                        if image_postprocess_fn is not None:
                            color = image_postprocess_fn(frame, color)
                        writer.append_data(color)
            elif video_format == "png":
                output_dir = os.path.join(os.path.dirname(video_path), video_sequence)
                os.makedirs(output_dir, exist_ok=True)
                for frame in range(num_frames):
                    render_frame_fn(frame)
                    color, _ = renderer.render(scene, flags=flags)
                    if image_postprocess_fn is not None:
                        color = image_postprocess_fn(frame, color)
                    cv2.imwrite(os.path.join(output_dir, str(frame).zfill(8) + ".png"), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
