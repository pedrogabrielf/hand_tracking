"""
Handtrack Perspective Canvas -- Step 2

Turns the 4 raw hand key points into a stable, ordered quadrilateral:

    - Orders the corners consistently (top-left, top-right, bottom-right,
      bottom-left) so downstream code (and, in Step 3, cv2.warpPerspective)
      always knows which corner is which.
    - Smooths corner positions frame-to-frame (landmarks are noisy, and a
      trembling quad outline looks bad / breaks user immersion).
    - Rejects degenerate quads (near-zero area) that would otherwise break
      the perspective warp in Step 3.
    - Draws the quad: outline, translucent fill preview, and corner labels.
"""

from typing import Optional

import cv2
import numpy as np

import config
from hand_tracker import HandLandmarksResult

CORNER_LABELS = ("TL", "TR", "BR", "BL")


class QuadrilateralTracker:
    """Builds a smoothed, consistently-ordered quadrilateral over time."""

    def __init__(self, smoothing_alpha: float = config.QUAD_SMOOTHING_ALPHA) -> None:
        self.smoothing_alpha = smoothing_alpha
        self._smoothed: Optional[np.ndarray] = None

    def update(self, key_points: HandLandmarksResult) -> Optional[np.ndarray]:
        """
        Feeds this frame's key points into the tracker.

        Returns a (4, 2) float32 array of smoothed corner coordinates, in
        `config.QUAD_CORNER_ORDER` order, or None if any of the 4 key
        points isn't currently detected (smoothing state is reset in that
        case, so the quad doesn't "snap" from a stale position once both
        hands reappear).
        """
        if not key_points.has_all_points():
            self._smoothed = None
            return None

        points_by_name = key_points.as_dict()
        raw = np.array(
            [points_by_name[name] for name in config.QUAD_CORNER_ORDER],
            dtype=np.float32,
        )

        if self._smoothed is None:
            self._smoothed = raw
        else:
            alpha = self.smoothing_alpha
            self._smoothed = alpha * raw + (1.0 - alpha) * self._smoothed

        return self._smoothed

    def reset(self) -> None:
        self._smoothed = None


def quad_area(corners: np.ndarray) -> float:
    """Signed-free polygon area in pixels^2 (via OpenCV's contour area)."""
    return float(cv2.contourArea(corners.astype(np.float32)))


def is_valid_quad(corners: np.ndarray, min_area: float = config.QUAD_MIN_AREA) -> bool:
    """Rejects degenerate quads (near-collinear points / near-zero area)."""
    return quad_area(corners) >= min_area


def draw_quadrilateral(
    frame: np.ndarray,
    corners: np.ndarray,
    fill_alpha: float = config.QUAD_FILL_ALPHA,
) -> None:
    """Draws the quad's translucent fill, outline, and corner labels."""
    pts = corners.astype(np.int32)

    if fill_alpha > 0:
        overlay = frame.copy()
        cv2.fillConvexPoly(overlay, pts, config.QUAD_FILL_COLOR)
        cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0, dst=frame)

    cv2.polylines(
        frame,
        [pts],
        isClosed=True,
        color=config.QUAD_LINE_COLOR,
        thickness=config.QUAD_LINE_THICKNESS,
        lineType=cv2.LINE_AA,
    )

    for label, (x, y) in zip(CORNER_LABELS, pts):
        cv2.putText(
            frame,
            label,
            (int(x) - 14, int(y) - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            config.QUAD_CORNER_LABEL_COLOR,
            2,
            cv2.LINE_AA,
        )
