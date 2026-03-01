#!/usr/bin/env python3
"""YOLOe Object Detector for Grasp Pipeline.

Uses YOLOe (text-prompt) to detect and segment objects in an RGB image.
Returns bounding boxes and segmentation masks for grasp planning.

Usage (standalone test):
    CUDA_VISIBLE_DEVICES=0 python examples/yoloe_grasp/grasp_pipeline/yoloe_detector.py \\
        --source examples/yoloe_grasp/grasp_pipeline/captured_data/color.png --names box
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class YOLOeDetector:
    """Wraps YOLOe for text-prompt object detection."""

    def __init__(self, checkpoint: str, device: str = "cuda:0"):
        """Initialize the YOLOe model.

        Args:
            checkpoint: Path to the YOLOe .pt checkpoint.
            device: Torch device string.
        """
        from ultralytics import YOLOE

        self.model = YOLOE(checkpoint)
        self.model.to(device)
        self.device = device
        self._current_names: list[str] | None = None
        logger.info(f"Loaded YOLOe: {checkpoint} on {device}")

    def detect(
        self,
        color_bgr: np.ndarray,
        names: list[str],
        conf_threshold: float = 0.25,
    ) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], float, str]]:
        """Detect the best-scoring object matching any of the given names.

        Args:
            color_bgr: BGR image (H, W, 3) uint8.
            names: List of class names to detect (text prompts).
            conf_threshold: Minimum confidence threshold.

        Returns:
            Tuple of (bbox_xyxy, mask, score, class_name) or None.
                bbox_xyxy: [x1, y1, x2, y2] in pixel coordinates.
                mask: (H, W) bool segmentation mask, or None if unavailable.
                score: Detection confidence.
                class_name: Detected class name.
        """
        from PIL import Image

        # Normalize names to list (YAML may parse single string)
        if isinstance(names, str):
            names = [names]

        # Set classes if changed
        if self._current_names != names:
            self.model.set_classes(names, self.model.get_text_pe(names))
            self._current_names = names
            logger.info(f"YOLOe classes set to: {names}")

        # Convert BGR → RGB PIL
        color_rgb = color_bgr[:, :, ::-1]
        pil_image = Image.fromarray(color_rgb)

        # Run inference
        results = self.model.predict(pil_image, verbose=False, conf=conf_threshold)
        result = results[0]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            logger.warning("YOLOe: No detections found")
            return None

        # Find best scoring detection
        confs = boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confs))

        bbox_xyxy = boxes.xyxy[best_idx].cpu().numpy()  # [x1, y1, x2, y2]
        score = float(confs[best_idx])
        cls_idx = int(boxes.cls[best_idx].cpu().numpy())
        class_name = names[cls_idx] if cls_idx < len(names) else f"class_{cls_idx}"

        # Extract segmentation mask if available
        mask = None
        if result.masks is not None and len(result.masks) > best_idx:
            import cv2
            from ultralytics.utils.ops import scale_image

            # result.masks.data is at model input resolution (e.g. 640x640).
            # For non-square images, YOLO applies letterbox padding.
            # scale_image() properly undoes this: crop padding → resize to orig.
            raw_mask = result.masks.data[best_idx].cpu().numpy()
            h_orig, w_orig = color_bgr.shape[:2]
            raw_shape = raw_mask.shape

            # scale_image expects (h, w) or (h, w, c)
            scaled = scale_image(raw_mask, (h_orig, w_orig))
            if scaled.ndim == 3:
                scaled = scaled[:, :, 0]
            mask = (scaled > 0.5).astype(bool)

            logger.info(f"Mask scaled: {raw_shape} → {mask.shape} "
                        f"(letterbox-aware)")
            logger.info(f"Mask pixels: {mask.sum()} / {mask.size} ({mask.sum()/mask.size*100:.1f}%)")

        logger.info(
            f"YOLOe detected '{class_name}' score={score:.3f} "
            f"bbox=[{bbox_xyxy[0]:.0f},{bbox_xyxy[1]:.0f},{bbox_xyxy[2]:.0f},{bbox_xyxy[3]:.0f}] "
            f"mask={'yes' if mask is not None else 'no'}"
        )

        return bbox_xyxy, mask, score, class_name


# ─── Standalone test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import cv2

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="YOLOe detector test")
    parser.add_argument("--source", required=True, help="Input image path")
    parser.add_argument("--names", nargs="+", default=["box"], help="Class names")
    parser.add_argument("--checkpoint", default="refence_code/yoloe/yoloe-v8l-seg.pt")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent.parent
    checkpoint = project_root / args.checkpoint

    detector = YOLOeDetector(str(checkpoint), device=args.device)

    img = cv2.imread(args.source)
    result = detector.detect(img, args.names)

    if result is None:
        print("No detections!")
    else:
        bbox, mask, score, name = result
        print(f"Detected: {name} ({score:.3f})")
        print(f"BBox: {bbox}")

        # Draw on image
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{name} {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if mask is not None:
            overlay = img.copy()
            overlay[mask] = (0, 255, 0)
            img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

        out = args.source.replace(".png", "-yoloe.jpg").replace(".jpg", "-yoloe.jpg")
        cv2.imwrite(out, img)
        print(f"Saved annotated image: {out}")
