#!/usr/bin/env python3
"""SAM3 Object Detector for Grasp Pipeline.

Uses SAM3 (text-prompt) to detect and segment objects in an RGB image.
Returns bounding boxes and segmentation masks for grasp planning —
same interface as YOLOeDetector.

Usage (standalone test):
    python examples/yoloe_grasp/grasp_pipeline/sam3_detector.py \\
        --source examples/yoloe_grasp/grasp_pipeline/captured_data/color.png --names box
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# SAM3 lives in refence_code/sam3 — add it to path at import time.
_SAM3_ROOT = Path(__file__).resolve().parents[3] / "refence_code" / "sam3"
if str(_SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAM3_ROOT))


class Sam3Detector:
    """Wraps SAM3 for text-prompt object detection and segmentation.

    Interface is identical to YOLOeDetector so the two can be used
    interchangeably in the grasp pipeline.
    """

    def __init__(self, device: str = "cuda:0"):
        """Load SAM3 model and processor.

        Args:
            device: Torch device string (e.g. 'cuda:0' or 'cpu').
        """
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.device = device
        logger.info("Loading SAM3 model...")
        model = build_sam3_image_model()
        self.processor = Sam3Processor(model)
        logger.info("SAM3 model loaded.")

    def detect(
        self,
        color_bgr: np.ndarray,
        names: list[str],
        conf_threshold: float = 0.0,
    ) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], float, str]]:
        """Detect the best-scoring object matching any of the given names.

        Runs one SAM3 inference per name and returns the highest-scoring
        detection across all prompts.

        Args:
            color_bgr: BGR image (H, W, 3) uint8.
            names: List of text prompts to try (e.g. ['box', 'rectangle']).
            conf_threshold: Minimum score threshold (0–1). SAM3 scores are
                            not calibrated probabilities; a value of 0.0
                            accepts all detections.

        Returns:
            Tuple of (bbox_xyxy, mask, score, class_name) or None.
                bbox_xyxy: np.ndarray [x1, y1, x2, y2] in pixel coordinates.
                mask:      (H, W) bool segmentation mask.
                score:     Detection confidence/score.
                class_name: The prompt string that produced this detection.
        """
        from PIL import Image

        if isinstance(names, str):
            names = [names]

        # Convert BGR → RGB PIL
        color_rgb = color_bgr[:, :, ::-1]
        pil_image = Image.fromarray(color_rgb)

        # Set image once (encoder pass)
        inference_state = self.processor.set_image(pil_image)

        best_bbox: np.ndarray | None = None
        best_mask: np.ndarray | None = None
        best_score: float = -1.0
        best_name: str = names[0]

        for name in names:
            try:
                output = self.processor.set_text_prompt(
                    state=inference_state, prompt=name
                )
            except Exception as e:
                logger.warning("SAM3 prompt '%s' failed: %s", name, e)
                continue

            boxes = output.get("boxes")
            scores = output.get("scores")
            masks = output.get("masks")

            if boxes is None or scores is None or len(scores) == 0:
                logger.warning("SAM3: no detections for prompt '%s'", name)
                continue

            import torch
            scores_np = (
                scores.cpu().numpy() if isinstance(scores, torch.Tensor)
                else np.asarray(scores)
            )
            best_idx = int(np.argmax(scores_np))
            top_score = float(scores_np[best_idx])

            if top_score < conf_threshold:
                logger.warning(
                    "SAM3: prompt '%s' top score %.3f below threshold %.3f",
                    name, top_score, conf_threshold,
                )
                continue

            if top_score > best_score:
                # Bounding box
                box = boxes[best_idx]
                bbox = (
                    box.cpu().numpy() if isinstance(box, torch.Tensor)
                    else np.asarray(box)
                ).astype(float)

                # Segmentation mask: shape (1, H, W) or (H, W)
                mask_np: np.ndarray | None = None
                if masks is not None and len(masks) > best_idx:
                    m = masks[best_idx]
                    if isinstance(m, torch.Tensor):
                        m = m.cpu().numpy()
                    if m.ndim == 3:          # (1, H, W)
                        m = m[0]
                    mask_np = m.astype(bool)

                best_bbox = bbox
                best_mask = mask_np
                best_score = top_score
                best_name = name

        if best_bbox is None:
            logger.warning("SAM3: no valid detections for prompts %s", names)
            return None

        logger.info(
            "SAM3 detected '%s' score=%.3f "
            "bbox=[%.0f,%.0f,%.0f,%.0f] mask=%s",
            best_name, best_score,
            best_bbox[0], best_bbox[1], best_bbox[2], best_bbox[3],
            "yes" if best_mask is not None else "no",
        )
        if best_mask is not None:
            logger.info(
                "Mask pixels: %d / %d (%.1f%%)",
                best_mask.sum(), best_mask.size,
                best_mask.sum() / best_mask.size * 100,
            )

        return best_bbox, best_mask, best_score, best_name


# ─── Standalone test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import cv2

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="SAM3 detector test")
    parser.add_argument("--source", required=True, help="Input image path")
    parser.add_argument("--names", nargs="+", default=["box"], help="Text prompts")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    detector = Sam3Detector(device=args.device)

    img = cv2.imread(args.source)
    result = detector.detect(img, args.names)

    if result is None:
        print("No detections!")
    else:
        bbox, mask, score, name = result
        print(f"Detected: {name} ({score:.3f})")
        print(f"BBox: {bbox}")

        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"SAM3: {name} {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if mask is not None:
            overlay = img.copy()
            overlay[mask] = (0, 0, 255)
            img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

        out = args.source.replace(".png", "-sam3.jpg").replace(".jpg", "-sam3.jpg")
        cv2.imwrite(out, img)
        print(f"Saved annotated image: {out}")
