"""
Handtrack Perspective Canvas -- Step 3

Projects a source image (or video frame) into the quadrilateral formed by
the 4 hand key points, using `cv2.getPerspectiveTransform` +
`cv2.warpPerspective`.

Two pieces:
    - `ImageSource` / `VideoSource`: supply the content to project (Step 3
      requirement: "an image or a video feed").
    - `PerspectiveWarper`: does the actual warp + compositing onto the
      camera frame.

Performance note: warping/compositing is restricted to the quadrilateral's
bounding box (plus a small margin for edge feathering) instead of the
full camera frame. Doing this over the whole 1280x720 frame every frame
would be wasteful when the hand-held quad usually covers a fraction of
the image -- this keeps Step 3 from eating into the >=30 FPS budget.
"""

from typing import Optional

import cv2
import numpy as np

import config


class ImageSource:
    """Serves the same static image every frame."""

    def __init__(self, path: str) -> None:
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(
                f"Could not load warp source image at '{path}'. "
                "Check config.WARP_SOURCE_IMAGE_PATH."
            )
        self._image = image

    def get_frame(self) -> np.ndarray:
        return self._image

    def release(self) -> None:
        pass


class VideoSource:
    """Serves consecutive frames from a video file, looping at the end."""

    def __init__(self, path: str) -> None:
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(
                f"Could not open warp source video at '{path}'. "
                "Check config.WARP_SOURCE_VIDEO_PATH."
            )
        self._last_frame: Optional[np.ndarray] = None

    def get_frame(self) -> np.ndarray:
        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        if ok:
            self._last_frame = frame
        # If even the looped read fails, fall back to the last good frame
        # (or a black placeholder) rather than crashing the main loop.
        if self._last_frame is None:
            self._last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return self._last_frame

    def release(self) -> None:
        self._cap.release()


def create_source():
    """Builds an ImageSource or VideoSource from config.WARP_SOURCE_TYPE."""
    if config.WARP_SOURCE_TYPE == "video":
        return VideoSource(config.WARP_SOURCE_VIDEO_PATH)
    return ImageSource(config.WARP_SOURCE_IMAGE_PATH)


class PerspectiveWarper:
    """Warps a source image into a quadrilateral on the destination frame."""

    def __init__(self, edge_feather_px: int = config.WARP_EDGE_FEATHER_PX) -> None:
        self.edge_feather_px = edge_feather_px

    def warp_into(
        self,
        frame: np.ndarray,
        source_img: np.ndarray,
        quad_corners: np.ndarray,
    ) -> None:
        """
        Warps `source_img` into `frame` in-place, filling the quadrilateral
        defined by `quad_corners` (4x2 float array, TL/TR/BR/BL order --
        see config.QUAD_CORNER_ORDER).
        """
        frame_h, frame_w = frame.shape[:2]
        dst_pts = quad_corners.astype(np.float32)

        margin = self.edge_feather_px + 2
        x0, y0, roi_w, roi_h = cv2.boundingRect(dst_pts.astype(np.int32))
        x0 = max(0, x0 - margin)
        y0 = max(0, y0 - margin)
        x1 = min(frame_w, x0 + roi_w + 2 * margin)
        y1 = min(frame_h, y0 + roi_h + 2 * margin)
        roi_w, roi_h = x1 - x0, y1 - y0
        if roi_w <= 0 or roi_h <= 0:
            return

        # Perspective transform mapping the source image's own corners to
        # the quad's corners, expressed relative to the ROI's origin.
        src_h, src_w = source_img.shape[:2]
        src_pts = np.array(
            [[0, 0], [src_w - 1, 0], [src_w - 1, src_h - 1], [0, src_h - 1]],
            dtype=np.float32,
        )
        dst_pts_roi = dst_pts - np.array([x0, y0], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(src_pts, dst_pts_roi)

        warped_roi = cv2.warpPerspective(
            source_img,
            transform,
            (roi_w, roi_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        mask_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.fillConvexPoly(mask_roi, dst_pts_roi.astype(np.int32), 255)
        if self.edge_feather_px > 0:
            k = 2 * self.edge_feather_px + 1
            mask_roi = cv2.GaussianBlur(mask_roi, (k, k), 0)

        alpha = (mask_roi.astype(np.float32) / 255.0)[:, :, None]
        frame_roi = frame[y0:y1, x0:x1]
        frame_roi[:] = (
            alpha * warped_roi.astype(np.float32)
            + (1.0 - alpha) * frame_roi.astype(np.float32)
        ).astype(np.uint8)
