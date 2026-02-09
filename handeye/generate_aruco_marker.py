#!/usr/bin/env python3
"""
ArUco Marker Generator

Generates a printable ArUco marker PNG image for hand-eye calibration.

Usage:
    python examples/generate_aruco_marker.py --marker_id 582 --marker_size 100

The generated marker will be saved as a PNG file that you can print.
Remember to measure the printed marker size in meters for calibration!
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ArUco dictionary mapping
ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


def generate_aruco_marker(
    marker_id: int,
    marker_size_px: int,
    dict_type: str,
    output_path: str,
    border_bits: int = 1,
) -> None:
    """
    Generate an ArUco marker and save as PNG.
    
    Args:
        marker_id: ID of the marker
        marker_size_px: Size of the marker in pixels
        dict_type: ArUco dictionary type
        output_path: Path to save the PNG
        border_bits: Width of white border in marker bits
    """
    if dict_type not in ARUCO_DICT_MAP:
        print(f"Error: Unknown dictionary type '{dict_type}'")
        print(f"Available: {', '.join(ARUCO_DICT_MAP.keys())}")
        sys.exit(1)
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[dict_type])
    
    # Check if marker_id is valid for this dictionary
    max_id = aruco_dict.bytesList.shape[0] - 1
    if marker_id > max_id:
        print(f"Error: Marker ID {marker_id} exceeds max {max_id} for {dict_type}")
        sys.exit(1)
    
    # Generate marker
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px)
    
    # Add white border for easier detection
    border_px = int(marker_size_px * 0.2)  # 20% border
    marker_with_border = cv2.copyMakeBorder(
        marker_img,
        border_px, border_px, border_px, border_px,
        cv2.BORDER_CONSTANT,
        value=255
    )
    
    # Save
    cv2.imwrite(output_path, marker_with_border)
    
    total_size = marker_size_px + 2 * border_px
    print(f"ArUco marker generated successfully!")
    print(f"  File: {output_path}")
    print(f"  ID: {marker_id}")
    print(f"  Dictionary: {dict_type}")
    print(f"  Size: {total_size}x{total_size} pixels")
    print()
    print("=" * 50)
    print("IMPORTANT: After printing, measure the BLACK SQUARE")
    print("(not including white border) with a ruler!")
    print()
    print("Recommended printed size: 5-10 cm")
    print()
    print("For calibration, use --marker_size in METERS:")
    print("  5 cm  -> --marker_size 0.05")
    print("  7 cm  -> --marker_size 0.07")
    print("  10 cm -> --marker_size 0.10")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Generate ArUco marker for hand-eye calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python examples/generate_aruco_marker.py --marker_id 582 --size 400

After printing, measure the black square (without white border) with a ruler.
Use this measurement in meters for --marker_size in handeye_calibration.py.
        """
    )
    parser.add_argument("--marker_id", type=int, default=42, help="Marker ID (default: 42)")
    parser.add_argument("--size", type=int, default=400, help="Marker size in pixels (default: 400)")
    parser.add_argument("--dict_type", type=str, default="DICT_4X4_50",
                        choices=list(ARUCO_DICT_MAP.keys()),
                        help="ArUco dictionary type (default: DICT_4X4_50)")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path (default: aruco_ID.png)")
    args = parser.parse_args()
    
    output_path = args.output or f"aruco_{args.marker_id}.png"
    
    generate_aruco_marker(
        marker_id=args.marker_id,
        marker_size_px=args.size,
        dict_type=args.dict_type,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
