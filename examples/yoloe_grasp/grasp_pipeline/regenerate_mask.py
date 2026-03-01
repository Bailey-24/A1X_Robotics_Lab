#!/usr/bin/env python3
"""
Regenerate workspace_mask.png for saved data using current config.yaml settings.

Usage:
    python examples/yoloe_grasp/grasp_pipeline/regenerate_mask.py

This allows you to tweak crop parameters in config.yaml and immediately see
the result in 'captured_data/workspace_mask.png' without running the robot.
"""
import sys
import yaml
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from examples.yoloe_grasp.grasp_pipeline.capture_rgbd import generate_workspace_mask

def main():
    config_path = PROJECT_ROOT / "grasp_pipeline" / "config.yaml"
    data_dir = PROJECT_ROOT / "grasp_pipeline" / "captured_data"
    
    if not data_dir.exists():
        print(f"Error: No captured data found at {data_dir}")
        sys.exit(1)
        
    print(f"Loading config from {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    ws_cfg = cfg["workspace"]
    print("Workspace config:")
    for k, v in ws_cfg.items():
        print(f"  {k}: {v}")
        
    # Load depth image to confirm dimensions
    depth_path = data_dir / "depth.png"
    if not depth_path.exists():
        print("Error: depth.png not found")
        sys.exit(1)
        
    depth = np.array(Image.open(str(depth_path)))
    h, w = depth.shape
    print(f"Loaded depth image: {w}x{h}")
    
    # Generate mask
    mask = generate_workspace_mask(
        height=h,
        width=w,
        mask_type=ws_cfg.get("mask_type", "upper_half"),
        crop_lr_pixels=ws_cfg.get("crop_lr_pixels", 100),
        crop_top=ws_cfg.get("crop_top", 0),
        crop_bottom=ws_cfg.get("crop_bottom", 0),
        crop_left=ws_cfg.get("crop_left", 0),
        crop_right=ws_cfg.get("crop_right", 0),
        depth_image=depth, # Pass depth if needed for depth_range, but 'custom' doesn't strictly need it
        min_depth_m=ws_cfg.get("min_depth", 0.1),
        max_depth_m=ws_cfg.get("max_depth", 0.6),
    )
    
    # Save it
    out_path = data_dir / "workspace_mask.png"
    cv2.imwrite(str(out_path), mask)
    print(f"\n✓ Regenerated and saved mask to: {out_path}")
    
    # Visualize if possible (requires GUI)
    try:
        current_mask = cv2.imread(str(out_path), cv2.IMREAD_GRAYSCALE)
        color_path = data_dir / "color.png"
        if color_path.exists():
            color = cv2.imread(str(color_path))
            # Overlay mask
            overlay = color.copy()
            overlay[current_mask == 0] = overlay[current_mask == 0] // 4 # Darken masked areas
            
            # Draw boundary lines
            if ws_cfg.get("mask_type") == "custom":
                # Draw red rectangles for cropped areas
                cl = ws_cfg.get("crop_left", 0)
                cr = ws_cfg.get("crop_right", 0)
                cb = ws_cfg.get("crop_bottom", 0)
                ct = ws_cfg.get("crop_top", 0)
                
                # Visualize crops
                if cl > 0: cv2.rectangle(overlay, (0,0), (cl, h), (0,0,255), 2)
                if cr > 0: cv2.rectangle(overlay, (w-cr,0), (w, h), (0,0,255), 2)
                if cb > 0: cv2.rectangle(overlay, (0, h-cb), (w, h), (0,0,255), 2)
                if ct > 0: cv2.rectangle(overlay, (0,0), (w, ct), (0,0,255), 2)
                
            out_vis = data_dir / "mask_visualization.jpg"
            cv2.imwrite(str(out_vis), overlay)
            print(f"  Visualization saved to: {out_vis}")
    except Exception as e:
        print(f"Debug visualization failed: {e}")

if __name__ == "__main__":
    main()
