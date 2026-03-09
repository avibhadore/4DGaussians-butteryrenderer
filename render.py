#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and educational use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# ── Modifications ────────────────────────────────────────────────────────────
# - mmcv replaced with importlib
# - Added --dense_video: renders new interpolated cameras between training poses
#   using cubic spline (translation) + RotationSpline (rotation) on SO(3).
#   Produces total_frames new cameras at 30fps for smooth video.
# ─────────────────────────────────────────────────────────────────────────────
import imageio
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from scene.cameras import MiniCam
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline
from time import time
import concurrent.futures


def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)

to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)


def build_interpolated_cameras(views, total_frames):
    """
    Generate total_frames new cameras by fitting smooth splines through
    all training camera poses, then sampling densely.

    - Translations: scipy CubicSpline
    - Rotations: scipy RotationSpline (smooth on SO(3))
    - Time: uniform 0 → 1

    This creates genuinely new camera poses between the training cameras,
    not just the training cameras repeated.
    """
    n = len(views)
    t_knots = np.linspace(0.0, 1.0, n)

    # Fit splines through all training poses
    translations = np.stack([v.T for v in views])           # (n, 3)
    rotations    = Rotation.from_matrix([v.R for v in views])
    trans_spline = CubicSpline(t_knots, translations)
    rot_spline   = RotationSpline(t_knots, rotations)

    # Time values (for 4DGS deformation)
    times = np.array([
        v.time if hasattr(v, 'time') else float(i) / (n - 1)
        for i, v in enumerate(views)
    ])
    time_spline = CubicSpline(t_knots, times)

    # Sample densely
    t_dense     = np.linspace(0.0, 1.0, total_frames)
    T_dense     = trans_spline(t_dense)
    R_dense     = rot_spline(t_dense).as_matrix()
    times_dense = np.clip(time_spline(t_dense), 0.0, 1.0)

    ref = views[0]
    result = []
    for i in range(total_frames):
        world_view = torch.tensor(
            getWorld2View2(R_dense[i], T_dense[i])
        ).transpose(0, 1).cuda()

        proj = getProjectionMatrix(
            znear=ref.znear, zfar=ref.zfar,
            fovX=ref.FoVx,  fovY=ref.FoVy
        ).transpose(0, 1).cuda()

        full_proj = (world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)

        result.append(MiniCam(
            width=ref.image_width,
            height=ref.image_height,
            fovy=ref.FoVy,
            fovx=ref.FoVx,
            znear=ref.znear,
            zfar=ref.zfar,
            world_view_transform=world_view,
            full_proj_transform=full_proj,
            time=float(times_dense[i]),
        ))

    return result


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, cam_type, fps=30):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path    = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path,    exist_ok=True)

    render_images = []
    gt_list       = []
    render_list   = []

    print("point nums:", gaussians._xyz.shape[0])
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0: time1 = time()
        rendering = render(view, gaussians, pipeline, background, cam_type=cam_type)["render"]
        render_images.append(to8b(rendering).transpose(1, 2, 0))
        render_list.append(rendering)
        if name in ["train", "test"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt = view['image'].cuda()
            gt_list.append(gt)

    time2 = time()
    print("FPS:", (len(views) - 1) / (time2 - time1))

    multithread_write(gt_list,     gts_path)
    multithread_write(render_list, render_path)

    out_path = os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4')
    imageio.mimwrite(out_path, render_images, fps=fps)
    print(f"  Written: {out_path}  ({len(render_images)} frames @ {fps}fps = {len(render_images)/fps:.1f}s)")


def render_sets(dataset, hyperparam, iteration, pipeline, skip_train, skip_test,
                skip_video, dense_video, total_frames, video_fps):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene     = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type  = scene.dataset_type

        bg_color   = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter,
                       scene.getTrainCameras(), gaussians, pipeline, background, cam_type, fps=video_fps)
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter,
                       scene.getTestCameras(), gaussians, pipeline, background, cam_type, fps=video_fps)
        if not skip_video:
            video_cams = scene.getVideoCameras()
            if dense_video:
                if total_frames is None:
                    total_frames = len(video_cams) * FRAMES_PER_CAM
                print(f"\n  Generating {total_frames} interpolated cameras from "
                      f"{len(video_cams)} training poses at {video_fps}fps "
                      f"= {total_frames/video_fps:.1f}s")
                video_cams = build_interpolated_cameras(video_cams, total_frames)
            render_set(dataset.model_path, "video", scene.loaded_iter,
                       video_cams, gaussians, pipeline, background, cam_type, fps=video_fps)


# ── Hardcoded rendering settings ─────────────────────────────────────────────
VIDEO_FPS         = 30
FRAMES_PER_CAM    = 15   # 30fps output / 2fps input = 15 frames per training camera
CONFIGS           = "arguments/hypernerf/default.py"
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model      = ModelParams(parser, sentinel=True)
    pipeline   = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet",     action="store_true")

    args = get_combined_args(parser)
    print("Rendering", args.model_path)

    # Load hardcoded config
    import importlib.util
    from utils.params_utils import merge_hparams
    spec = importlib.util.spec_from_file_location("config", CONFIGS)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = {k: v for k, v in vars(config_module).items() if not k.startswith("_")}
    args = merge_hparams(args, config)

    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        hyperparam.extract(args),
        args.iteration,
        pipeline.extract(args),
        skip_train=True,
        skip_test=True,
        skip_video=False,
        dense_video=True,
        total_frames=None,   # computed dynamically from num cameras * FRAMES_PER_CAM
        video_fps=VIDEO_FPS,
    )
