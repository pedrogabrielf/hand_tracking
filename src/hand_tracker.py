"""
Thin wrapper around MediaPipe's HandLandmarker task that exposes exactly
what this project needs: the four key fingertip points used to build the
perspective quadrilateral (left/right thumb tips and left/right index
fingertips).

MediaPipe 1.0 removed the legacy `mp.solutions.hands` API in favor of the
Tasks API (`mediapipe.tasks.python.vision.HandLandmarker`), which this
module wraps.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

import config

Point = Tuple[int, int]

_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS


@dataclass
class HandLandmarksResult:
    """Pixel-space key points detected in a single frame."""

    left_thumb: Optional[Point] = None
    left_index: Optional[Point] = None
    right_thumb: Optional[Point] = None
    right_index: Optional[Point] = None

    def as_dict(self) -> Dict[str, Optional[Point]]:
        return {
            "left_thumb": self.left_thumb,
            "left_index": self.left_index,
            "right_thumb": self.right_thumb,
            "right_index": self.right_index,
        }

    def has_all_points(self) -> bool:
        return all(v is not None for v in self.as_dict().values())


class HandTracker:
    """Detects hands in a BGR frame and extracts the four key landmarks."""

    def __init__(
        self,
        model_path: str = config.HAND_LANDMARKER_MODEL_PATH,
        max_num_hands: int = config.MAX_NUM_HANDS,
        min_hand_detection_confidence: float = config.MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence: float = config.MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE,
    ) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"HandLandmarker model not found at '{model_path}'.\n"
                f"Download it first, e.g.:\n"
                f'  curl -sSL -o "{model_path}" "{config.HAND_LANDMARKER_MODEL_URL}"'
            )

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

    def process(self, frame_rgb) -> Tuple[HandLandmarksResult, object]:
        """
        Runs HandLandmarker on an RGB frame (video-mode, monotonically
        increasing timestamps) and returns a tuple of (key_points, raw_result).

        `frame_rgb` must already be the RGB image the caller wants
        handedness computed against (typically the mirrored/selfie-view
        frame -- see config.MIRROR_VIEW).

        `raw_result` is returned so callers can optionally draw the full
        skeleton via `draw_skeleton` without re-running inference.
        """
        height, width = frame_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # VIDEO mode requires strictly increasing timestamps; using a
        # monotonic counter avoids relying on wall-clock precision.
        self._frame_timestamp_ms += 1
        result = self._landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        key_points = HandLandmarksResult()
        if not result.hand_landmarks or not result.handedness:
            return key_points, result

        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            label = handedness[0].category_name  # "Left" or "Right"

            thumb_lm = landmarks[config.LANDMARK_THUMB_TIP]
            index_lm = landmarks[config.LANDMARK_INDEX_TIP]

            thumb_px = (int(thumb_lm.x * width), int(thumb_lm.y * height))
            index_px = (int(index_lm.x * width), int(index_lm.y * height))

            if label == "Left":
                key_points.left_thumb = thumb_px
                key_points.left_index = index_px
            else:
                key_points.right_thumb = thumb_px
                key_points.right_index = index_px

        return key_points, result

    def draw_skeleton(self, frame_bgr, hand_landmarks_list) -> None:
        """Draws the full 21-point hand skeleton (optional, for debugging)."""
        for landmarks in hand_landmarks_list or []:
            vision.drawing_utils.draw_landmarks(frame_bgr, landmarks, _CONNECTIONS)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
