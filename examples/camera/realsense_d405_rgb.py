#!/usr/bin/env python3
"""Intel RealSense D405 RGB + Depth Viewer.

This script captures and displays both RGB and depth frames from an Intel
RealSense D405 camera using pyrealsense2 and OpenCV. The depth image is
colorized with a JET colormap and displayed side-by-side with the RGB image.

Usage:
    python examples/camera/realsense_d405_rgb.py

Controls:
    - Press 'q' or ESC to quit
    - Press 's' to save current frame as PNG (both RGB and depth)
    - Press 'v' to toggle video recording
    - Press 'c' to cycle depth colormap
"""
from __future__ import annotations

import sys
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print("Error: pyrealsense2 not installed.")
    print("Install with: pip install pyrealsense2")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Available depth colormaps
DEPTH_COLORMAPS = [
    ("JET", cv2.COLORMAP_JET),
    ("TURBO", cv2.COLORMAP_TURBO),
    ("INFERNO", cv2.COLORMAP_INFERNO),
    ("HOT", cv2.COLORMAP_HOT),
    ("BONE", cv2.COLORMAP_BONE),
]


def find_realsense_device() -> rs.device | None:
    """Find and return a RealSense device with color capability.
    
    Returns:
        RealSense device object or None if not found.
    """
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        logger.error("No RealSense devices detected")
        return None
    
    for device in devices:
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        product_line = device.get_info(rs.camera_info.product_line)
        
        logger.info(f"Found device: {name} (Serial: {serial}, Product: {product_line})")
        
        sensors = list(device.sensors)
        for s in sensors:
            sensor_name = s.get_info(rs.camera_info.name)
            logger.info(f"  Sensor: {sensor_name}")
        
        # D405 uses "Stereo Module" which provides both depth and color
        # Other D400 series have a separate "RGB Camera" sensor
        has_color_capability = any(
            s.get_info(rs.camera_info.name) in ('RGB Camera', 'Stereo Module')
            for s in sensors
        )
        
        if has_color_capability:
            logger.info("✓ Color-capable sensor available")
            return device
        else:
            logger.warning("✗ No color-capable sensor on this device")
    
    return None


def main() -> int:
    """Main function to capture and display RGB + Depth frames from D405.
    
    Returns:
        Exit code (0 for success, 1 for failure).
    """
    device = find_realsense_device()
    if device is None:
        return 1
    
    # Configure pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable color stream - 640x480 @ 15fps (Matched with depth for stability)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    # Enable depth stream - 640x480 @ 15fps
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    
    # Note: To use 1280x720, set both streams to 5fps.
    
    # Start streaming
    try:
        profile = pipeline.start(config)
        logger.info("Pipeline started successfully")
        
        # Get actual stream parameters
        color_profile = profile.get_stream(rs.stream.color)
        color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        logger.info(f"Color stream: {color_intrinsics.width}x{color_intrinsics.height}")
        
        depth_profile = profile.get_stream(rs.stream.depth)
        depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        logger.info(f"Depth stream: {depth_intrinsics.width}x{depth_intrinsics.height}")
        
        # Get depth scale for distance calculation
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        logger.info(f"Depth scale: {depth_scale} meters/unit")
        
    except RuntimeError as e:
        logger.error(f"Failed to start pipeline: {e}")
        return 1
    
    # Create align object to align depth to color frame
    align = rs.align(rs.stream.color)
    
    # Create output directory for saved frames and videos
    output_dir = Path(__file__).parent / "captured_frames"
    output_dir.mkdir(exist_ok=True)
    
    logger.info("Streaming RGB + Depth... Press 'q' to quit, 's' to save, 'v' to record, 'c' to change colormap")
    
    frame_count = 0
    is_recording = False
    video_writer = None
    recording_output_path = None
    colormap_idx = 0  # Current colormap index
    
    try:
        while True:
            # Wait for frames (with timeout)
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                logger.warning("Frame timeout - camera may be disconnected")
                continue
            
            # Align depth to color
            aligned_frames = align.process(frames)
            
            # Get color and depth frames
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            
            frame_count += 1
            
            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Colorize depth image for visualization
            # Normalize depth to 0-255 range for colormap
            # D405 typical range: 0.07m - 0.5m (70mm - 500mm)
            depth_colormap_name, depth_colormap = DEPTH_COLORMAPS[colormap_idx]
            depth_colorized = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                depth_colormap
            )
            
            # Add frame info overlay
            timestamp = color_frame.get_timestamp()
            info_text = f"Frame: {frame_count} | Time: {timestamp:.0f}ms"
            
            # Get depth at center point for display
            center_x, center_y = color_image.shape[1] // 2, color_image.shape[0] // 2
            center_depth = depth_frame.get_distance(center_x, center_y)
            depth_info = f"Center depth: {center_depth:.3f}m | Colormap: {depth_colormap_name}"
            
            # Create display copies
            display_color = color_image.copy()
            display_depth = depth_colorized.copy()
            
            # Draw crosshair at center on both images
            cross_size = 15
            cv2.drawMarker(display_color, (center_x, center_y), (0, 255, 0),
                           cv2.MARKER_CROSS, cross_size, 1)
            cv2.drawMarker(display_depth, (center_x, center_y), (255, 255, 255),
                           cv2.MARKER_CROSS, cross_size, 1)
            
            # Add text overlays
            cv2.putText(display_color, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_color, "RGB", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.putText(display_depth, depth_info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_depth, "DEPTH", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Stack images side by side
            combined_image = np.hstack((display_color, display_depth))
            
            # Record frame if recording is active
            if is_recording and video_writer is not None:
                # Record the raw combined (without REC indicator)
                raw_combined = np.hstack((color_image, depth_colorized))
                video_writer.write(raw_combined)
                # Visual indicator for recording
                cv2.circle(combined_image, (combined_image.shape[1] - 30, 30), 10, (0, 0, 255), -1)
                cv2.putText(
                    combined_image, "REC", (combined_image.shape[1] - 80, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )
            
            # Display the combined frame
            cv2.imshow('RealSense D405 RGB + Depth', combined_image)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                logger.info("Quit requested")
                break
            elif key == ord('s'):  # Save frames
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Save RGB
                rgb_filename = output_dir / f"d405_rgb_{timestamp_str}.png"
                cv2.imwrite(str(rgb_filename), color_image)
                # Save raw depth as 16-bit PNG (preserves actual depth values)
                depth_filename = output_dir / f"d405_depth_{timestamp_str}.png"
                cv2.imwrite(str(depth_filename), depth_image)
                # Save colorized depth
                depth_color_filename = output_dir / f"d405_depth_color_{timestamp_str}.png"
                cv2.imwrite(str(depth_color_filename), depth_colorized)
                # Save combined view
                combined_filename = output_dir / f"d405_combined_{timestamp_str}.png"
                cv2.imwrite(str(combined_filename), combined_image)
                logger.info(f"Frames saved: {rgb_filename.name}, {depth_filename.name}, {combined_filename.name}")
            elif key == ord('c'):  # Cycle colormap
                colormap_idx = (colormap_idx + 1) % len(DEPTH_COLORMAPS)
                logger.info(f"Depth colormap: {DEPTH_COLORMAPS[colormap_idx][0]}")
            elif key == ord('v'):  # Toggle recording
                if is_recording:
                    # Stop recording
                    is_recording = False
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                    logger.info(f"Recording stopped. Saved to {recording_output_path}")
                else:
                    # Start recording
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    recording_output_path = output_dir / f"d405_rgbd_video_{timestamp_str}.mp4"
                    
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = 15.0  # Match camera fps
                    frame_size = (combined_image.shape[1], combined_image.shape[0])
                    
                    video_writer = cv2.VideoWriter(str(recording_output_path), fourcc, fps, frame_size)
                    
                    if video_writer.isOpened():
                        is_recording = True
                        logger.info(f"Recording started: {recording_output_path}")
                    else:
                        logger.error("Failed to start video recording")
                        video_writer = None
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Cleanup
        if video_writer:
            video_writer.release()
        pipeline.stop()
        cv2.destroyAllWindows()
        logger.info(f"Streaming stopped. Total frames captured: {frame_count}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

