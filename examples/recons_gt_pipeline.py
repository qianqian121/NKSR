# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import glob
import multiprocessing
import os.path

import click
import nksr
import torch
from joblib import Parallel, delayed

from pycg import vis, exp
from pathlib import Path
import numpy as np
from common import load_waymo_example, warning_on_low_memory

import open3d as o3d


def load_pcd():
    dataset = './result/'
    out_dir = dataset
    pcd_file = os.path.join(dataset, 'opt_scan_knn_combined.pcd')
    pcd = o3d.io.read_point_cloud(pcd_file)
    xyz_np = np.array(pcd.points)
    normals_np = np.array(pcd.normals)
    return xyz_np, normals_np


def create_scan_from_file(filepath):
    pcd = o3d.io.read_point_cloud(filepath)

    xyz_np = np.array(pcd.points)
    normals_np = np.array(pcd.normals)
    return xyz_np, normals_np


def load_pcds(dataset):
    num_cores = multiprocessing.cpu_count()
    # dataset = './result/'
    out_dir = dataset
    pcd_path = os.path.join(dataset, 'opt_global_motion_pcdbin')
    scan_names = sorted(glob.glob(os.path.join(pcd_path, "*.pcd")), reverse=False)
    num_scans = len(scan_names)
    # scan_names = scan_names[:600]
    # scan_names = scan_names[:300]
    # scan_names = scan_names[:200]
    # scan_names = scan_names[200:300]
    scan_names = scan_names[300:350]
    # scan_names = scan_names[:100]
    # scan_names = scan_names[:20]
    all_np_scans_tuples = Parallel(n_jobs=num_cores)(
        # delayed(read_cloud_from_pcd)(filepath) for filepath in iter_filepaths)
        delayed(create_scan_from_file)(filepath) for filepath in scan_names)
    all_xyz_np, all_normals_np = zip(*all_np_scans_tuples)
    xyz_np = np.concatenate([xyz for xyz in all_xyz_np])
    normals_np = np.concatenate([xyz for xyz in all_normals_np])
    return xyz_np, normals_np


def load_pcd_file(pcd_file):
    pcd = o3d.io.read_point_cloud(pcd_file)
    xyz_np = np.array(pcd.points)
    normals_np = np.array(pcd.normals)
    return xyz_np, normals_np


@click.command()
@click.option(
    "--dataset",
    "-d",
    type=click.Path(exists=True),
    default=os.environ["HOME"] + "/lidar/result",
    help="Location of the gt pipeline result dataset",
)
@click.option(
    "--out_dir",
    "-o",
    type=click.Path(exists=False),
    default="./lidar/result",
    help="Where to store the results",
)
def main_clean(dataset, out_dir):
    out_dir = dataset

    warning_on_low_memory(20000.0)
    xyz_np, normals_np = load_pcd_file(dataset)
    # xyz_np, normals_np = load_pcd()
    # xyz_np, normals_np = load_pcds(dataset)

    device = torch.device("cuda:0")
    reconstructor = nksr.Reconstructor(device)
    reconstructor.chunk_tmp_device = torch.device("cpu")

    input_xyz = torch.from_numpy(xyz_np).float().to(device)
    input_normal = torch.from_numpy(normals_np).float().to(device)

    field = reconstructor.reconstruct(
        input_xyz, normal=input_normal, detail_level=None,
        # input_xyz, normal=input_normal, detail_level=1.0,
        # input_xyz, normal=input_normal, detail_level=0.99014332,
        # input_xyz, normal=input_normal, detail_level=0.9901,
        # input_xyz, normal=input_normal, detail_level=0.99,
        # input_xyz, normal=input_normal, detail_level=0.95,
        # input_xyz, normal=input_normal, detail_level=0.85,
        # Minor configs for better efficiency (not necessary)
        approx_kernel_grad=True, solver_tol=1e-4, fused_mode=True, 
        # Chunked reconstruction (if OOM)
        chunk_size=168
        # chunk_size=160
        # chunk_size=128
        # chunk_size=51.2
        # chunk_size=25.6
        # chunk_size=12.8
    )
    
    # (Optional) Convert to CPU for mesh extraction
    # field.to_("cpu")
    # reconstructor.network.to("cpu")

    mesh = field.extract_dual_mesh(mise_iter=1)
    mesh = vis.mesh(mesh.v, mesh.f)

    vis.show_3d([mesh], [vis.pointcloud(xyz_np)])


if __name__ == '__main__':
    main_clean()
