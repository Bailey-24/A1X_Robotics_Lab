#!/usr/bin/env python3
"""Intel RealSense D405 RGB Camera Reader.

This script captures and displays RGB frames from an Intel RealSense D405 camera
using pyrealsense2 and OpenCV.

Usage:
    python examples/realsense_d405_rgb.py

Controls:
    - Press 'q' or ESC to quit
    - Press 's' to save current frame as PNG
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
    """Main function to capture and display RGB frames from D405.
    
    Returns:
        Exit code (0 for success, 1 for failure).
    """
    device = find_realsense_device()
    if device is None:
        return 1
    
    # Configure pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable color stream - D405 supports up to 1280x720 @ 30fps for color
    # Using 640x480 @ 30fps for lower latency
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    try:
        profile = pipeline.start(config)
        logger.info("Pipeline started successfully")
        
        # Get actual stream parameters
        color_profile = profile.get_stream(rs.stream.color)
        intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        logger.info(f"Color stream: {intrinsics.width}x{intrinsics.height}")
        
    except RuntimeError as e:
        logger.error(f"Failed to start pipeline: {e}")
        return 1
    
    # Create output directory for saved frames
    output_dir = Path(__file__).parent / "captured_frames"
    output_dir.mkdir(exist_ok=True)
    
    logger.info("Streaming RGB frames... Press 'q' to quit, 's' to save frame")
    
    frame_count = 0
    
    try:
        while True:
            # Wait for frames (with timeout)
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                logger.warning("Frame timeout - camera may be disconnected")
                continue
            
            # Get color frame
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            frame_count += 1
            
            # Convert to numpy array (BGR format for OpenCV)
            color_image = np.asanyarray(color_frame.get_data())
            
            # Add frame info overlay
            timestamp = color_frame.get_timestamp()
            info_text = f"Frame: {frame_count} | Time: {timestamp:.0f}ms"
            cv2.putText(
                color_image, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            
            # Display the frame
            cv2.imshow('RealSense D405 RGB', color_image)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                logger.info("Quit requested")
                break
            elif key == ord('s'):  # Save frame
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = output_dir / f"d405_frame_{timestamp_str}.png"
                cv2.imwrite(str(filename), color_image)
                logger.info(f"Frame saved to {filename}")
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Cleanup
        pipeline.stop()
        cv2.destroyAllWindows()
        logger.info(f"Streaming stopped. Total frames captured: {frame_count}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
