import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import base64
import sys
import yaml
from pathlib import Path
from ollama import chat

def get_prompt():
    if len(sys.argv) > 1:
        return sys.argv[1]
    else:
        return "What is in this image?"

def image_to_base64(image, ext=".jpg", quality=95):
    """
    Convert OpenCV image to Base64 string
    :param image: OpenCV image (numpy array)
    :param ext: image extension (e.g. .jpg/.png)
    :param quality: JPG quality (0-100)
    :return: Base64 string
    """
    try:
        encode_param = [int(cv.IMWRITE_JPEG_QUALITY), quality] if ext == ".jpg" else []
        retval, img_buffer = cv.imencode(ext, image, encode_param)
        if not retval:
            raise ValueError("Image encoding failed")

        base64_str = base64.b64encode(img_buffer).decode("utf-8")
        return base64_str
    except Exception as e:
        print(f"Failed to convert to Base64: {e}")
        return None

def load_camera_config():
    """
    Load camera parameters from examples/yoloe_grasp/config.yaml
    """
    config_path = Path(__file__).parent.parent / "yoloe_grasp" / "config.yaml"
    try:
        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)
            return full_config.get("camera", {})
    except Exception as e:
        print(f"Warning: Failed to load config from {config_path}. Using default params. Error: {e}")
        return {"width": 640, "height": 480, "fps": 15}

def main():
    prompt = get_prompt()
    camera_cfg = load_camera_config()
    
    width = camera_cfg.get("width", 640)
    height = camera_cfg.get("height", 480)
    fps = camera_cfg.get("fps", 15)

    print(f"Camera Config - resolution: {width}x{height}, fps: {fps}")
    print(f"Using text prompt: {prompt}")

    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable color stream using parameters from yaml
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    try:
        print("Starting camera pipeline...")
        pipeline.start(config)

        # Wait for camera auto-exposure to stabilize
        stable_frames = min(fps * 2, 60)  # usually wait about 2 seconds
        print(f"Waiting for auto-exposure to stabilize (discarding {stable_frames} frames)...")
        for _ in range(stable_frames):
            pipeline.wait_for_frames()

        print("Capturing frame...")
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if color_frame:
            color_img = np.asanyarray(color_frame.get_data())

            # Convert to base64 for inference
            base64_result = image_to_base64(color_img)
            if base64_result:
                print("Sending request to qwen3.5...")
                try:
                    response = chat(
                        model="qwen3.5",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                                "images": [base64_result],
                            }
                        ],
                    )
                    print("\nResponse:")
                    print(response.message.content)
                except Exception as e:
                    print(f"Inference error: {e}")
        else:
            print("Failed to capture image frame")

    except Exception as e:
        print(f"Camera error: {e}")
    finally:
        pipeline.stop()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
