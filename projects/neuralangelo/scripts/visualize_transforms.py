"""
=============================
@author: Shandong
@email: shandong.wang@intel.com
@time: 2024/1/8:上午9:06
@IDE: PyCharm
=============================
"""

# Set the work directory to the imaginaire root.
import os, sys, time
import pathlib

root_dir = pathlib.Path().absolute().parents[2]
os.chdir(root_dir)
print(f"Root Directory Path: {root_dir}")

# Import Python libraries.
import numpy as np
import torch
import k3d
import json
from collections import OrderedDict
# Import imaginaire modules.
from projects.nerf.utils import camera, visualize
from third_party.colmap.scripts.python.read_write_model import read_model

# Read the COLMAP data.
colmap_path = "datasets/lego_ds2"
json_fname = f"{colmap_path}/transforms.json"
with open(json_fname) as file:
    meta = json.load(file)
center = meta["sphere_center"]
radius = meta["sphere_radius"]
# Convert camera poses.
poses = []
for frame in meta["frames"]:
    c2w = torch.tensor(frame["transform_matrix"])
    c2w[:, 1:3] *= -1  # from opengl to original
    w2c = c2w.inverse()
    pose = w2c[:3]  # [3,4]
    poses.append(pose)
poses = torch.stack(poses, dim=0)
print(f"# images: {len(poses)}")

vis_depth = 0.2
k3d_textures = []

# (optional) visualize the images.
# This block can be skipped if we don't want to visualize the image observations.
for i, frame in enumerate(meta["frames"]):
    image_fname = frame["file_path"]
    image_path = f"{colmap_path}/{image_fname}"
    with open(image_path, "rb") as file:
        binary = file.read()
    # Compute the corresponding image corners in 3D.
    pose = poses[i]
    corners = torch.tensor([[-0.5, 0.5, 1], [0.5, 0.5, 1], [-0.5, -0.5, 1]])
    corners *= vis_depth
    corners = camera.cam2world(corners, pose)
    puv = [corners[0].tolist(), (corners[1] - corners[0]).tolist(), (corners[2] - corners[0]).tolist()]
    k3d_texture = k3d.texture(binary, file_format="jpg", puv=puv)
    k3d_textures.append(k3d_texture)

# Visualize the bounding sphere.
json_fname = f"{colmap_path}/transforms.json"
with open(json_fname) as file:
    meta = json.load(file)
center = meta["sphere_center"]
radius = meta["sphere_radius"]
# ------------------------------------------------------------------------------------
# These variables can be adjusted to make the bounding sphere fit the region of interest.
# The adjusted values can then be set in the config as data.readjust.center and data.readjust.scale
readjust_center = np.array([0., 0., 0.])
readjust_scale = 1.
# ------------------------------------------------------------------------------------
center += readjust_center
radius *= readjust_scale
# Make some points to hallucinate a bounding sphere.
sphere_points = np.random.randn(100000, 3)
sphere_points = sphere_points / np.linalg.norm(sphere_points, axis=-1, keepdims=True)
sphere_points = sphere_points * radius + center

# Visualize with K3D.
plot = k3d.plot(name="poses", height=800, camera_rotate_speed=5.0, camera_zoom_speed=3.0, camera_pan_speed=1.0)
k3d_objects = visualize.k3d_visualize_pose(poses, vis_depth=vis_depth, xyz_length=0.02, center_size=0.01,
                                           xyz_width=0.005, mesh_opacity=0.)
for k3d_object in k3d_objects:
    plot += k3d_object
for k3d_texture in k3d_textures:
    plot += k3d_texture
plot += k3d.points(sphere_points, color=0x4488ff, point_size=0.01, shader="flat")
plot.display()
plot.camera_fov = 30.0
