#!/usr/bin/env python3
"""Grasp Prediction Module for Grasp Pipeline.

Wraps GraspNet-baseline model for 6-DOF grasp prediction from RGBD input.
Outputs grasp poses (rotation + translation) in camera coordinate frame.

Usage (standalone test):
    CUDA_VISIBLE_DEVICES=0 python examples/yoloe_grasp/grasp_pipeline/grasp_predictor.py
"""
from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class GraspPredictor:
    """Wraps GraspNet-baseline for grasp pose prediction.

    Loads the model once and provides a predict() method for repeated inference.
    All outputs are in the camera coordinate frame.
    """

    def __init__(
        self,
        checkpoint_path: str,
        graspnet_root: str,
        num_point: int = 20000,
        num_view: int = 300,
        collision_thresh: float = 0.01,
        voxel_size: float = 0.01,
        max_grasps: int = 50,
    ):
        """Initialize the GraspNet model.

        Args:
            checkpoint_path: Path to the trained model checkpoint (.tar file).
            graspnet_root: Path to the graspnet-baseline code directory.
            num_point: Number of points to sample from point cloud.
            num_view: Number of views for the model.
            collision_thresh: Threshold for collision detection.
            voxel_size: Voxel size for collision detection preprocessing.
            max_grasps: Maximum number of grasps to return after NMS.
        """
        self.checkpoint_path = str(checkpoint_path)
        self.graspnet_root = str(graspnet_root)
        self.num_point = num_point
        self.num_view = num_view
        self.collision_thresh = collision_thresh
        self.voxel_size = voxel_size
        self.max_grasps = max_grasps

        # Add GraspNet paths to sys.path
        for subdir in ["models", "dataset", "utils"]:
            path = os.path.join(self.graspnet_root, subdir)
            if path not in sys.path:
                sys.path.insert(0, path)
        if self.graspnet_root not in sys.path:
            sys.path.insert(0, self.graspnet_root)

        # Lazy imports — only import when actually constructing
        import torch
        from graspnet import GraspNet, pred_decode
        from collision_detector import ModelFreeCollisionDetector
        from data_utils import CameraInfo, create_point_cloud_from_depth_image

        self._torch = torch
        self._pred_decode = pred_decode
        self._ModelFreeCollisionDetector = ModelFreeCollisionDetector
        self._CameraInfo = CameraInfo
        self._create_point_cloud_from_depth_image = create_point_cloud_from_depth_image

        # Also need GraspGroup from graspnetAPI
        from graspnetAPI import GraspGroup
        self._GraspGroup = GraspGroup

        # Load model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.net = GraspNet(
            input_feature_dim=0,
            num_view=self.num_view,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
        )
        self.net.to(self.device)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.net.eval()

        logger.info(
            f"Loaded GraspNet checkpoint: {self.checkpoint_path} "
            f"(epoch {checkpoint['epoch']})"
        )

    def predict(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        intrinsic_matrix: np.ndarray,
        factor_depth: int = 1000,
        workspace_mask: Optional[np.ndarray] = None,
        min_depth_m: float = 0.1,
        max_depth_m: float = 0.6,
    ) -> Tuple[Optional[object], Optional[object]]:
        """Predict grasp poses from RGBD input.

        Args:
            color_image: BGR image (H, W, 3) uint8.
            depth_image: Depth image (H, W) uint16, in mm.
            intrinsic_matrix: Camera intrinsic matrix (3, 3).
            factor_depth: Depth scale factor (1000 = mm).
            workspace_mask: Optional binary mask (H, W) uint8. If None, generated
                from depth range.
            min_depth_m: Minimum depth for workspace mask (meters).
            max_depth_m: Maximum depth for workspace mask (meters).

        Returns:
            Tuple of (grasp_group, point_cloud):
                - grasp_group: GraspGroup with sorted grasps (best first), or None
                - point_cloud: Open3D point cloud of the scene, or None
        """
        import open3d as o3d

        torch = self._torch

        # --- Prepare color (float32, 0-1 range) ---
        # GraspNet demo reads with PIL (RGB), but we have BGR from OpenCV
        color_rgb = color_image[:, :, ::-1]  # BGR -> RGB
        color_float = color_rgb.astype(np.float32) / 255.0

        # --- Generate workspace mask if not provided ---
        if workspace_mask is None:
            depth_m = depth_image.astype(np.float32) / factor_depth
            workspace_mask = (
                (depth_image > 0)
                & (depth_m >= min_depth_m)
                & (depth_m <= max_depth_m)
            ).astype(np.uint8) * 255

        # --- Generate point cloud from depth ---
        fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
        cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
        height, width = depth_image.shape[:2]

        camera = self._CameraInfo(
            float(width), float(height), fx, fy, cx, cy, factor_depth
        )
        cloud = self._create_point_cloud_from_depth_image(depth_image, camera, organized=True)

        # Debug: Check point cloud scale
        valid_z = cloud[depth_image > 0][:, 2]
        if len(valid_z) > 0:
            min_z, max_z = valid_z.min(), valid_z.max()
            logger.info(f"Point cloud Z range: [{min_z:.4f}, {max_z:.4f}]")
            if max_z > 3.0:
                logger.warning(f"⚠️  Max Z > 3.0 ({max_z:.2f})! Point cloud might be in millimeters.")
                logger.warning(f"    Check factor_depth (current: {factor_depth})")
        else:
            logger.warning("Point cloud has no valid depth points!")

        # --- Mask and sample points ---
        mask = (workspace_mask > 0) & (depth_image > 0)
        cloud_masked = cloud[mask]
        color_masked = color_float[mask]

        if len(cloud_masked) == 0:
            logger.warning("No valid points in workspace mask — cannot predict grasps")
            return None, None

        logger.info(f"Valid points in workspace: {len(cloud_masked)}")

        # Sample to fixed number of points
        if len(cloud_masked) >= self.num_point:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(
                len(cloud_masked), self.num_point - len(cloud_masked), replace=True
            )
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # --- Build Open3D point cloud for collision detection ---
        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        o3d_cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

        # --- Prepare model input ---
        end_points = {}
        cloud_tensor = torch.from_numpy(
            cloud_sampled[np.newaxis].astype(np.float32)
        ).to(self.device)
        end_points["point_clouds"] = cloud_tensor
        end_points["cloud_colors"] = color_sampled

        # --- Forward pass ---
        logger.info("Running GraspNet inference...")
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = self._pred_decode(end_points)

        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = self._GraspGroup(gg_array)
        logger.info(f"Raw predictions: {len(gg)} grasps")

        # --- Collision detection ---
        if self.collision_thresh > 0 and len(gg) > 0:
            mfc = self._ModelFreeCollisionDetector(
                np.array(o3d_cloud.points), voxel_size=self.voxel_size
            )
            collision_mask = mfc.detect(
                gg, approach_dist=0.05, collision_thresh=self.collision_thresh
            )
            gg = gg[~collision_mask]
            logger.info(f"After collision filtering: {len(gg)} grasps")

        if len(gg) == 0:
            logger.warning("No valid grasps after collision filtering")
            return None, o3d_cloud

        # --- NMS and sort ---
        gg.nms()
        gg.sort_by_score()
        gg = gg[: self.max_grasps]
        logger.info(f"After NMS + top-{self.max_grasps}: {len(gg)} grasps")

        return gg, o3d_cloud

    def get_best_grasp(
        self, gg
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """Extract the best grasp from a GraspGroup.

        Args:
            gg: GraspGroup instance (sorted by score, best first).

        Returns:
            Tuple of (rotation_matrix, translation, score) or None.
                - rotation_matrix: (3, 3) grasp orientation in camera frame
                - translation: (3,) grasp position in camera frame [meters]
                - score: confidence score
        """
        if gg is None or len(gg) == 0:
            return None

        best = gg[0]
        return best.rotation_matrix, best.translation, best.score


# ─── Standalone test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    import yaml

    project_root = Path(__file__).parent.parent.parent.parent
    config_path = Path(__file__).parent / "config.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    gn_cfg = cfg["graspnet"]
    checkpoint = project_root / gn_cfg["checkpoint_path"]
    graspnet_root = project_root / gn_cfg["graspnet_root"]

    print("GraspNet Predictor Test")
    print("=" * 40)
    print(f"Checkpoint: {checkpoint}")
    print(f"GraspNet root: {graspnet_root}")

    predictor = GraspPredictor(
        checkpoint_path=str(checkpoint),
        graspnet_root=str(graspnet_root),
        num_point=gn_cfg["num_point"],
        num_view=gn_cfg["num_view"],
        collision_thresh=gn_cfg["collision_thresh"],
        voxel_size=gn_cfg["voxel_size"],
        max_grasps=gn_cfg["max_grasps"],
    )

    # Test with example data
    example_dir = graspnet_root / "doc" / "example_data"
    if example_dir.exists():
        from PIL import Image
        import scipy.io as scio

        color = np.array(Image.open(str(example_dir / "color.png")))[:, :, ::-1]  # RGB->BGR
        depth = np.array(Image.open(str(example_dir / "depth.png")))
        meta = scio.loadmat(str(example_dir / "meta.mat"))
        intrinsic = meta["intrinsic_matrix"]

        gg, cloud = predictor.predict(color, depth, intrinsic)
        if gg is not None:
            result = predictor.get_best_grasp(gg)
            if result:
                rot, trans, score = result
                print(f"\nBest grasp:")
                print(f"  Score: {score:.4f}")
                print(f"  Translation: {trans}")
                print(f"  Rotation:\n{rot}")
        else:
            print("No grasps predicted")
    else:
        print(f"Example data not found at {example_dir}")
