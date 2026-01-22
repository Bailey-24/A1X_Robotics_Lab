from asmagic import ARDataSubscriber

# Create subscriber with your iPhone's IP address
sub = ARDataSubscriber("192.168.31.13")

try:
    # Continuous data streaming
    for data in sub:
        # All sensor data in one frame
        print(f"Timestamp: {data.timestamp}")
        print(f"Velocity: {data.velocity}")
        print(f"Local Pose: {data.local_pose}")
        print(f"Global Pose: {data.global_pose}")
        print(f"Camera Intrinsics: {data.camera_intrinsics}")

        # Access image data
        if data.has_color_image:
            # Color: bytes(jpeg format) or array
            color_bytes = data.color_bytes
            color_array = data.color_array  # or shortcut: data.color
            print(f"Color: {len(color_bytes)} bytes, array shape: {color_array.shape}")

        if data.has_depth_image:
            # Depth: Numpy array
            depth = data.depth_array  # or shortcut: data.depth
            print(f"Depth: {depth.shape}")

except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    sub.close()