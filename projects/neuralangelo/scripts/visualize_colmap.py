"""
=============================
@author: Shandong
@email: shandong.wang@intel.com
@time: 2024/1/2:下午2:02
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
import plotly.graph_objs as go
from collections import OrderedDict
# Import imaginaire modules.
from projects.nerf.utils import camera, visualize
from third_party.colmap.scripts.python.read_write_model import read_model

# Read the COLMAP data.
colmap_path = "datasets/lego_ds2"
cameras, images, points_3D = read_model(path=f"{colmap_path}/sparse", ext=".bin")
# Convert camera poses.
images = OrderedDict(sorted(images.items()))
qvecs = torch.from_numpy(np.stack([image.qvec for image in images.values()]))
tvecs = torch.from_numpy(np.stack([image.tvec for image in images.values()]))
Rs = camera.quaternion.q_to_R(qvecs)
poses = torch.cat([Rs, tvecs[..., None]], dim=-1)  # [N,3,4]
print(f"# images: {len(poses)}")
# Get the sparse 3D points and the colors.
xyzs = torch.from_numpy(np.stack([point.xyz for point in points_3D.values()]))
rgbs = np.stack([point.rgb for point in points_3D.values()])
rgbs_int32 = (rgbs[:, 0] * 2**16 + rgbs[:, 1] * 2**8 + rgbs[:, 2]).astype(np.uint32)
print(f"# points: {len(xyzs)}")

vis_depth = 0.2

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

# You can choose to visualize with Plotly...
x, y, z = *xyzs.T,
colors = rgbs / 255.0
sphere_x, sphere_y, sphere_z = *sphere_points.T,
sphere_colors = ["#4488ff"] * len(sphere_points)
traces_poses = visualize.plotly_visualize_pose(poses, vis_depth=vis_depth, xyz_length=0.02, center_size=0.01, xyz_width=0.005, mesh_opacity=0.05)
trace_points = go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=dict(size=1, color=colors, opacity=1), hoverinfo="skip")
trace_sphere = go.Scatter3d(x=sphere_x, y=sphere_y, z=sphere_z, mode="markers", marker=dict(size=0.5, color=sphere_colors, opacity=0.7), hoverinfo="skip")
# traces_all = traces_poses + [trace_points, trace_sphere]
traces_all = traces_poses  + [trace_sphere]
layout = go.Layout(scene=dict(xaxis=dict(showspikes=False, backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(0,0,0,0.1)"),
                              yaxis=dict(showspikes=False, backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(0,0,0,0.1)"),
                              zaxis=dict(showspikes=False, backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(0,0,0,0.1)"),
                              xaxis_title="X", yaxis_title="Y", zaxis_title="Z", dragmode="orbit",
                              aspectratio=dict(x=1, y=1, z=1), aspectmode="data"), height=800)
fig = go.Figure(data=traces_all, layout=layout)
fig.show()

# ... or visualize with K3D.
plot = k3d.plot(name="poses", height=800, camera_rotate_speed=5.0, camera_zoom_speed=3.0, camera_pan_speed=1.0)
k3d_objects = visualize.k3d_visualize_pose(poses, vis_depth=vis_depth, xyz_length=0.02, center_size=0.01, xyz_width=0.005, mesh_opacity=0.05)
for k3d_object in k3d_objects:
    plot += k3d_object
plot += k3d.points(xyzs, colors=rgbs_int32, point_size=0.02, shader="flat")
plot += k3d.points(sphere_points, color=0x4488ff, point_size=0.01, shader="flat")
plot.display()
plot.camera_fov = 30.0

