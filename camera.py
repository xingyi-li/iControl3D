import os
import json
import base64
import numpy as np
import gradio as gr
from PIL import Image
import io
import torch
import torchvision
import torch.nn as nn
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
)
from pytorch3d.transforms import euler_angles_to_matrix
from scipy.ndimage import median_filter
from envmap import EnvironmentMap

from third_party.DPT.run_monodepth import run_dpt_online
from third_party.ZoeDepth.run_monodepth import run_zoe_online
import global_var
from lib.mesh_fusion.util import get_pinhole_intrinsics_from_fov
from lib.utils.utils import save_image, pil_to_torch, save_poisson_mesh

def base64_to_numpy(base64_str):
    data = base64.b64decode(str(base64_str))
    ret = np.frombuffer(data, dtype=np.float32)
    return ret

def numpy_to_base64(arr):
    base64_bytes = base64.b64encode(arr)
    base64_str = base64_bytes.decode("ascii")
    return base64_str


class PointsRenderer(nn.Module):
    """
    A class for rendering a batch of points. The class should
    be initialized with a rasterizer and compositor class which each have a forward
    function.
    """

    def __init__(self, rasterizer, compositor) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def to(self, device):
        # Manually move to device rasterizer as the cameras
        # within the class are not of type nn.Module
        self.rasterizer = self.rasterizer.to(device)
        self.compositor = self.compositor.to(device)
        return self

    def forward(self, point_clouds, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)

        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius

        if type(r) == torch.Tensor:
            if r.shape[-1] > 1:
                idx = fragments.idx.clone()
                idx[idx == -1] = 0
                r = r[:, idx.squeeze().long()]
                r = r.permute(0, 3, 1, 2)

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        return images


def create_pcd_renderer(h, w, intrinsics, R=None, T=None, radius=None, device="cuda"):
    # Initialize a camera.
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    if R is None:
        R = torch.eye(3)[None]  # (1, 3, 3)
    if T is None:
        T = torch.zeros(1, 3)  # (1, 3)
    cameras = PerspectiveCameras(R=R, T=T,
                                 device=device,
                                 focal_length=((-fx, -fy),),
                                 principal_point=(tuple(intrinsics[:2, -1]),),
                                 image_size=((h, w),),
                                 in_ndc=False,
                                 )

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. Refer to raster_points.py for explanations of these parameters.
    if radius is None:
        radius = 1.5 / min(h, w) * 2.0

    raster_settings = PointsRasterizationSettings(
        image_size=(h, w),
        radius=radius,
        points_per_pixel=8,
    )

    # Create a points renderer by compositing points using an alpha compositor (nearer points
    # are weighted more heavily). See [1] for an explanation.
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )
    return rasterizer, renderer


def homogenize_pt(coord):
    return torch.cat([coord, torch.ones_like(coord[..., :1])], dim=-1)

def unproject_pts_pt(intrinsics, coords, depth):
    if coords.shape[-1] == 2:
        coords = homogenize_pt(coords)
    intrinsics = intrinsics.squeeze()[:3, :3]
    coords = torch.inverse(intrinsics).mm(coords.T) * depth.reshape(1, -1)
    return coords.T   # [n, 3]

def remove_noise_in_dpt_disparity(disparity, kernel_size=5):
    return median_filter(disparity, size=kernel_size)


def get_coord_grids_pt(h, w, device, homogeneous=False):
    """
    create pxiel coordinate grid
    :param h: height
    :param w: weight
    :param device: device
    :param homogeneous: if homogeneous coordinate
    :return: coordinates [h, w, 2]
    """
    y = torch.arange(0, h).to(device)
    x = torch.arange(0, w).to(device)
    grid_y, grid_x = torch.meshgrid(y, x)
    if homogeneous:
        return torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1)
    return torch.stack([grid_x, grid_y], dim=-1)  # [h, w, 2]


def update_pcd(
        buffer_str,
        intrinsic_str,
        pcd_output_state,
        rgb_output_state,
):
    pcd = global_var.get_value("pcd")
    rgb = global_var.get_value("rgb")

    to_tensor = torchvision.transforms.ToTensor()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    data = base64.b64decode(str(buffer_str))
    pil = Image.open(io.BytesIO(data))
    buffer = np.array(pil)
    pcd = torch.from_numpy(pcd.reshape(-1, 3)).float().to(device)
    rgb = torch.from_numpy(rgb.reshape(-1, 4)).float().to(device)
    print(pcd.shape)
    intrinsic = torch.from_numpy(base64_to_numpy(intrinsic_str).reshape(3, 3)).float().to(device)
    w, h = intrinsic[:2, -1] * 2
    print(w, h)
    print("buffer:", buffer.shape)

    dpt_model_path = "third_party/DPT/weights/dpt_hybrid-midas-501f0c75.pt"

    src_depth = run_zoe_online(input_img=buffer[..., :3])

    buffer = to_tensor(buffer).float()[None].to(device)


    print("buffer:", buffer.shape)
    mask = (buffer[:, -1] > 0.5).reshape(-1, 1)
    mask = mask.repeat(1, 3)
    print("mask:", mask.shape)
    print("src_depth:", src_depth.shape)
    coord = get_coord_grids_pt(h, w, device=device).float()[None]
    pts = unproject_pts_pt(intrinsic, coord.reshape(-1, 2), src_depth)
    pts = pts[mask].reshape(-1, 3)
    print("pts:", pts.shape)

    buffer_reshaped = buffer.permute(0, 2, 3, 1).reshape(-1, 4)

    mask = mask[:, 0:1].repeat(1, 4)

    pcd = torch.cat([pcd, pts], dim=0)
    rgb = torch.cat([rgb, buffer_reshaped[mask].reshape(-1, 4)], dim=0)
    print("pcd:", pcd.shape)
    print("rgb:", rgb.shape)
    print("pcd.numpy:", pcd.cpu().numpy().shape)
    print("rgb.numpy:", rgb.cpu().numpy().shape)

    pcd = pcd.cpu().numpy()
    rgb = rgb.cpu().numpy()

    global_var.set_value("pcd", pcd)
    global_var.set_value("rgb", rgb)

    return (
        gr.update(label=str(pcd_output_state + 1), value=str(pcd_output_state + 1)),
        gr.update(label=str(rgb_output_state + 1), value=str(rgb_output_state + 1)),
        pcd_output_state + 1,
        rgb_output_state + 1,
    )


def update_3d(buffer_str):
    data = base64.b64decode(str(buffer_str))
    pil = Image.open(io.BytesIO(data))
    buffer = np.array(pil)
    H, W, _ = buffer.shape

    pipeline = global_var.get_value("mesh_pipeline")
    if not pipeline.initialized:
        pipeline.offset = 0
        pipeline.H = H
        pipeline.W = W
        pipeline.rendered_depth = torch.zeros((H, W), device=pipeline.args.device)  # depth rendered from point cloud
        pipeline.inpaint_mask = torch.ones((H, W), device=pipeline.args.device, dtype=torch.bool)  # 1: no projected points (need to be inpainted) | 0: have projected points
        # pipeline.alpha_map = torch.tensor(buffer[:, :, -1]/255 > 0.5, device=pipeline.args.device, dtype=torch.bool)
        pipeline.K = get_pinhole_intrinsics_from_fov(H=H, W=W, fov_in_degrees=pipeline.args.fov).to(pipeline.world_to_cam)
        pipeline.nerf_transforms_dict = pipeline.build_nerf_transforms_header()

        # environment map
        # pipeline.envmap = EnvironmentMap(H*2, format_="latlong")
        pipeline.initialized = True

        pipeline.setup_start_image(pil, offset=pipeline.offset)

        # save image, depth and pose
        filename = save_image(pipeline.current_image_pil, "rendering", pipeline.offset,
                              pipeline.args.output_rendering_path)
        # the depths are assumed to be 16-bit or 32-bit and to be in millimeters
        # set the depth of sky to be zero (means invalid)

        if pipeline.sky_mask is not None:
            sky_mask = pipeline.sky_mask.squeeze().detach().cpu().numpy()
            predicted_depth = pipeline.predicted_depth.squeeze().detach().cpu().numpy()
            depth = np.where(sky_mask, 0., predicted_depth) * 1000.0
        else:
            depth = pipeline.predicted_depth.squeeze().detach().cpu().numpy() * 1000.0

        filename_depth = save_image(
            Image.fromarray(depth.astype(np.uint16)),
            "depth", pipeline.offset, pipeline.args.output_depth_path)

        pipeline.save_poses(os.path.join(pipeline.args.out_path, "seen_poses.json"), pipeline.seen_poses)
        pipeline.append_nerf_extrinsic(os.path.basename(pipeline.args.output_rendering_path), filename,
                                       os.path.basename(pipeline.args.output_depth_path), filename_depth)
        pipeline.save_nerf_transforms()

    else:
        pipeline.offset += 1
        pipeline.seen_poses.append(pipeline.world_to_cam.clone())

        # update images
        pipeline.current_image_pil = pil
        pipeline.current_image = pil_to_torch(pil, pipeline.args.device)

        # predict depth, add to 3D structure
        pipeline.add_next_image(0, offset=pipeline.offset)

        # update bounding box
        pipeline.calc_bounding_box()

        if pipeline.args.clean_mesh_every_nth > 0 and (pipeline.offset) % pipeline.args.clean_mesh_every_nth == 0:
            pipeline.clean_mesh()

        # save image, depth and pose
        filename = save_image(pipeline.current_image_pil, "rendering", pipeline.offset,
                              pipeline.args.output_rendering_path)
        # the depths are assumed to be 16-bit or 32-bit and to be in millimeters
        # set the depth of sky to be zero (means invalid)

        if pipeline.sky_mask is not None:
            sky_mask = pipeline.sky_mask.squeeze().detach().cpu().numpy()
            predicted_depth = pipeline.predicted_depth.squeeze().detach().cpu().numpy()
            # depth = np.where(sky_mask, 0., predicted_depth) * 1000.0
            depth = np.where(sky_mask, 0., predicted_depth) * 1000.0
        else:
            depth = pipeline.predicted_depth.squeeze().detach().cpu().numpy() * 1000.0

        filename_depth = save_image(
            Image.fromarray(depth.astype(np.uint16)),
            "depth", pipeline.offset, pipeline.args.output_depth_path)

        pipeline.save_poses(os.path.join(pipeline.args.out_path, "seen_poses.json"), pipeline.seen_poses)
        pipeline.append_nerf_extrinsic(os.path.basename(pipeline.args.output_rendering_path), filename,
                                       os.path.basename(pipeline.args.output_depth_path), filename_depth)
        pipeline.save_nerf_transforms()


    torch.cuda.empty_cache()


def camera_transform(
        transform,
        render_output_state,
):
    pipeline = global_var.get_value("mesh_pipeline")
    device = pipeline.args.device
    current_world_to_cam = pipeline.world_to_cam
    current_euler_angles = pipeline.euler_angles
    current_envmap_euler_angles = pipeline.envmap_euler_angles
    R1 = current_world_to_cam[:3, :3]
    T1 = current_world_to_cam[:3, 3]
    delta_rad = 0.1
    delta_dist = 0.1

    if transform == "rotate_left":
        new_euler_angles = torch.tensor([0.0, -delta_rad, 0.0]).to(device)
        now_euler_angles = current_euler_angles + new_euler_angles
        R = euler_angles_to_matrix(now_euler_angles, "XYZ").to(device)

        now_envmap_euler_angles = current_envmap_euler_angles + torch.tensor([0.0, -delta_rad, 0.0]).to(device)
        envmap_R = euler_angles_to_matrix(now_envmap_euler_angles, "XYZ").to(device)
        pipeline.envmap_euler_angles = now_envmap_euler_angles
        pipeline.envmap_R = envmap_R

        T = T1
        RT = torch.cat([R, T[:, None]], dim=-1).to(device)  # RT is [4,4]
        RT = torch.cat([RT, torch.tensor([[0, 0, 0, 1]]).to(RT)], dim=0)
        pipeline.world_to_cam = RT
        pipeline.euler_angles = now_euler_angles

    elif transform == "rotate_right":
        new_euler_angles = torch.tensor([0.0, delta_rad, 0.0]).to(device)
        now_euler_angles = current_euler_angles + new_euler_angles
        R = euler_angles_to_matrix(now_euler_angles, "XYZ").to(device)

        now_envmap_euler_angles = current_envmap_euler_angles + torch.tensor([0.0, delta_rad, 0.0]).to(device)
        envmap_R = euler_angles_to_matrix(now_envmap_euler_angles, "XYZ").to(device)
        pipeline.envmap_euler_angles = now_envmap_euler_angles
        pipeline.envmap_R = envmap_R

        T = T1
        RT = torch.cat([R, T[:, None]], dim=-1).to(device)  # RT is [4,4]
        RT = torch.cat([RT, torch.tensor([[0, 0, 0, 1]]).to(RT)], dim=0)
        pipeline.world_to_cam = RT
        pipeline.euler_angles = now_euler_angles
    elif transform == "rotate_up":
        new_euler_angles = torch.tensor([delta_rad, 0.0, 0.0]).to(device)
        now_euler_angles = current_euler_angles + new_euler_angles
        R = euler_angles_to_matrix(now_euler_angles, "XYZ").to(device)

        now_envmap_euler_angles = current_envmap_euler_angles + torch.tensor([-delta_rad, 0.0, 0.0]).to(device)
        envmap_R = euler_angles_to_matrix(now_envmap_euler_angles, "XYZ").to(device)
        pipeline.envmap_euler_angles = now_envmap_euler_angles
        pipeline.envmap_R = envmap_R

        T = T1
        RT = torch.cat([R, T[:, None]], dim=-1).to(device)  # RT is [4,4]
        RT = torch.cat([RT, torch.tensor([[0, 0, 0, 1]]).to(RT)], dim=0)
        pipeline.world_to_cam = RT
        pipeline.euler_angles = now_euler_angles
    elif transform == "rotate_down":
        new_euler_angles = torch.tensor([-delta_rad, 0.0, 0.0]).to(device)
        now_euler_angles = current_euler_angles + new_euler_angles
        R = euler_angles_to_matrix(now_euler_angles, "XYZ").to(device)

        now_envmap_euler_angles = current_envmap_euler_angles + torch.tensor([delta_rad, 0.0, 0.0]).to(device)
        envmap_R = euler_angles_to_matrix(now_envmap_euler_angles, "XYZ").to(device)
        pipeline.envmap_euler_angles = now_envmap_euler_angles
        pipeline.envmap_R = envmap_R

        T = T1
        RT = torch.cat([R, T[:, None]], dim=-1).to(device)  # RT is [4,4]
        RT = torch.cat([RT, torch.tensor([[0, 0, 0, 1]]).to(RT)], dim=0)
        pipeline.world_to_cam = RT
        pipeline.euler_angles = now_euler_angles
    elif transform == "move_forward":
        R = R1
        T = T1 + torch.tensor([0.0, 0.0, -delta_dist]).to(device)
        RT = torch.cat([R, T[:, None]], dim=-1).to(device)  # RT is [4,4]
        RT = torch.cat([RT, torch.tensor([[0, 0, 0, 1]]).to(RT)], dim=0)
        pipeline.world_to_cam = RT
    elif transform == "move_backward":
        R = R1
        T = T1 + torch.tensor([0.0, 0.0, delta_dist]).to(device)
        RT = torch.cat([R, T[:, None]], dim=-1).to(device)  # RT is [4,4]
        RT = torch.cat([RT, torch.tensor([[0, 0, 0, 1]]).to(RT)], dim=0)
        pipeline.world_to_cam = RT
    elif transform == "move_left":
        R = R1
        T = T1 + torch.tensor([-delta_dist, 0.0, 0.0]).to(device)
        RT = torch.cat([R, T[:, None]], dim=-1).to(device)  # RT is [4,4]
        RT = torch.cat([RT, torch.tensor([[0, 0, 0, 1]]).to(RT)], dim=0)
        pipeline.world_to_cam = RT
    elif transform == "move_right":
        R = R1
        T = T1 + torch.tensor([delta_dist, 0.0, 0.0]).to(device)
        RT = torch.cat([R, T[:, None]], dim=-1).to(device)  # RT is [4,4]
        RT = torch.cat([RT, torch.tensor([[0, 0, 0, 1]]).to(RT)], dim=0)
        pipeline.world_to_cam = RT
    elif transform == "move_up":
        R = R1
        T = T1 + torch.tensor([0.0, -delta_dist, 0.0]).to(device)
        RT = torch.cat([R, T[:, None]], dim=-1).to(device)  # RT is [4,4]
        RT = torch.cat([RT, torch.tensor([[0, 0, 0, 1]]).to(RT)], dim=0)
        pipeline.world_to_cam = RT
    elif transform == "move_down":
        R = R1
        T = T1 + torch.tensor([0.0, delta_dist, 0.0]).to(device)
        RT = torch.cat([R, T[:, None]], dim=-1).to(device)  # RT is [4,4]
        RT = torch.cat([RT, torch.tensor([[0, 0, 0, 1]]).to(RT)], dim=0)
        pipeline.world_to_cam = RT
    else:
        raise NotImplementedError

    _, rendered_image_pil, inpaint_mask_pil = pipeline.project()

    if pipeline.envmap is not None:
        R = pipeline.envmap_R.cpu().numpy()
        sky_crop = pipeline.envmap.project(vfov=pipeline.args.fov,
                                           rotation_matrix=R,
                                           ar=pipeline.W / pipeline.H,
                                           resolution=(pipeline.W, pipeline.H),
                                           projection="perspective",
                                           mode="normal")
        sky_crop_mask = pipeline.envmap_mask.project(vfov=pipeline.args.fov,
                                                     rotation_matrix=R,
                                                     ar=pipeline.W / pipeline.H,
                                                     resolution=(pipeline.W, pipeline.H),
                                                     projection="perspective",
                                                     mode="normal")

        sky_crop = (np.array(sky_crop) * 255).astype(np.uint8)
        sky_crop_mask = ((sky_crop_mask > 0.5) * 255).astype(np.uint8)
        render_output = np.array(rendered_image_pil)
        inpaint_mask = np.array(inpaint_mask_pil)

        sky_crop[~inpaint_mask.astype(np.bool)] = render_output[~inpaint_mask.astype(np.bool)]
        render_output = sky_crop
        inpaint_mask = ~np.logical_or(~inpaint_mask, sky_crop_mask)
        inpaint_mask = inpaint_mask.astype(np.uint8) * 255
    else:
        render_output = np.array(rendered_image_pil)
        inpaint_mask = np.array(inpaint_mask_pil)

    render_output = np.concatenate([render_output, ~inpaint_mask[:, :, :1]], axis=2)
    out_pil = Image.fromarray(render_output)
    out_buffer = io.BytesIO()
    out_pil.save(out_buffer, format="PNG")
    out_buffer.seek(0)
    base64_bytes = base64.b64encode(out_buffer.read())
    render_output_str = base64_bytes.decode("ascii")

    return (
        gr.update(label=str(render_output_state + 1), value=render_output_str),
        render_output_state + 1,
    )


def scene_completion():
    pipeline = global_var.get_value("mesh_pipeline")

    # load trajectories
    trajectories = json.load(open(pipeline.args.trajectory_file, "r"))

    # check if there is a custom prompt in the first trajectory
    # would use it to generate start image, if we have to
    if "prompt" in trajectories[0]:
        pipeline.args.prompt = trajectories[0]["prompt"]

    # generate using all trajectories
    offset = pipeline.offset + 1
    for t in trajectories:
        pipeline.set_trajectory(t)
        offset = pipeline.generate_images(offset=offset)

    # save outputs before completion
    pipeline.clean_mesh()
    # intermediate_mesh_path = pipeline.save_mesh("after_generation.ply")
    # save_poisson_mesh(intermediate_mesh_path, depth=pipeline.args.poisson_depth, max_faces=pipeline.args.max_faces_for_poisson)

    # run completion
    # pipeline.args.update_mask_after_improvement = True
    # pipeline.complete_mesh(offset=offset)
    # pipeline.clean_mesh()

    # Now no longer need the models
    pipeline.remove_models()

    # save outputs after completion
    # final_mesh_path = pipeline.save_mesh()

    # # run poisson mesh reconstruction
    # mesh_poisson_path = save_poisson_mesh(final_mesh_path, depth=args.poisson_depth, max_faces=args.max_faces_for_poisson)
    # pipeline.save_mesh("after_generation.ply")
    # save_poisson_mesh(final_mesh_path, depth=pipeline.args.poisson_depth,
    #                   max_faces=pipeline.args.max_faces_for_poisson)

    print("Finished. Outputs stored in:", pipeline.args.out_path)
