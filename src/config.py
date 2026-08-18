"""
Central configuration for the Handtrack Perspective Canvas project.

Keeping every tunable value in one place makes it easy to adjust behavior
(camera setup, detection thresholds, UI options) without touching the
core logic in later steps.
"""

import os

# --- Camera -----------------------------------------------------------------
# NOTE (macOS): "Continuity Camera" can register a nearby iPhone as a video
# device over AVFoundation. Its position in the device list is NOT stable
# across runs (it can shift ahead of or behind the Mac's built-in camera
# depending on whether the iPhone is awake/connected), so a hardcoded index
# alone is unreliable.
#
# On macOS, `main.py` resolves the camera at startup via
# camera_utils.find_camera_index, which matches by hardware DEVICE TYPE
# (AVCaptureDeviceTypeBuiltIn*) -- not by localized name. Name matching is
# unreliable: the built-in camera's name changes with the OS language
# (e.g. "Câmera FaceTime HD" in pt-BR) and the iPhone's device name is
# just the phone's own name (e.g. "iPhone de Pedro Gabriel Camera"), which
# never contains the word "Continuity".
#
# PREFERRED/EXCLUDE_CAMERA_NAME_SUBSTRINGS below are only used as a
# secondary tie-breaker (e.g. more than one built-in camera) or as a
# last-resort fallback if no built-in device type is found at all.
# CAMERA_INDEX is the final fallback, only used on non-macOS or if pyobjc
# isn't installed -- in that case run `python src/list_cameras.py` to find
# the correct index manually.
CAMERA_INDEX = 0
PREFERRED_CAMERA_NAME_SUBSTRINGS = ["FaceTime", "Integrated", "Built-in"]
EXCLUDE_CAMERA_NAME_SUBSTRINGS = ["iPhone", "iPad", "Continuity"]
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30

# If True, the frame is flipped horizontally before processing, producing a
# natural "mirror" view. MediaPipe's handedness output assumes a mirrored
# (selfie-style) input, so flipping first keeps "Left"/"Right" labels
# consistent with the user's own left/right hands.
MIRROR_VIEW = True

# --- MediaPipe Hands (Tasks API) --------------------------------------------
# MediaPipe 1.0 dropped the old `mp.solutions.hands` API in favor of the
# HandLandmarker task, which requires a downloaded model bundle (.task file).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAND_LANDMARKER_MODEL_PATH = os.path.join(
    _PROJECT_ROOT, "models", "hand_landmarker.task"
)
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

MAX_NUM_HANDS = 2
MIN_HAND_DETECTION_CONFIDENCE = 0.7
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.7

# Landmark indices we care about (see MediaPipe Hands landmark map).
LANDMARK_THUMB_TIP = 4
LANDMARK_INDEX_TIP = 8

# --- Display ---------------------------------------------------------------
WINDOW_NAME = "Handtrack Perspective Canvas"
SHOW_FULL_SKELETON = True  # draw all 21 landmarks/connections per hand
KEY_POINT_RADIUS = 10
KEY_POINT_COLORS = {
    "left_thumb": (255, 0, 255),    # magenta
    "left_index": (255, 255, 0),    # cyan-ish (BGR)
    "right_thumb": (0, 255, 255),   # yellow
    "right_index": (0, 255, 0),     # green
}

# --- Quadrilateral (Step 2) --------------------------------------------------
# Corner order used everywhere downstream (quad.py, and later the
# perspective warp in Step 3): top-left, top-right, bottom-right,
# bottom-left -- matching the natural "framing" gesture where both index
# fingers point up (top corners) and both thumbs point down (bottom
# corners), like a photographer framing a shot with their hands.
QUAD_CORNER_ORDER = ("left_index", "right_index", "right_thumb", "left_thumb")

# Exponential moving average factor applied to corner positions each frame:
# new = alpha * raw + (1 - alpha) * previous. Lower = smoother/more lag,
# higher = snappier/more jitter. Landmarks are noisy frame-to-frame, and a
# jittery quad outline looks bad, so we smooth it before drawing/warping.
QUAD_SMOOTHING_ALPHA = 0.4

# Quads smaller than this (in pixels^2) are treated as degenerate (hands
# too close together / nearly collinear points) and are not drawn -- a
# near-zero-area quad would also break cv2.getPerspectiveTransform later.
QUAD_MIN_AREA = 2000

QUAD_LINE_COLOR = (255, 255, 255)
QUAD_LINE_THICKNESS = 3
QUAD_FILL_COLOR = (255, 180, 0)  # translucent preview of the "canvas" area
QUAD_FILL_ALPHA = 0.18
QUAD_CORNER_LABEL_COLOR = (255, 255, 255)

# --- Perspective warp (Step 3) ----------------------------------------------
# What gets projected into the quadrilateral: a static image or a looping
# video file. Toggle at runtime with the 'w' key (see main.py).
WARP_SOURCE_TYPE = "image"  # "image" or "video"
WARP_SOURCE_IMAGE_PATH = os.path.join(_PROJECT_ROOT, "assets", "canvas_source.jpg")
WARP_SOURCE_VIDEO_PATH = os.path.join(_PROJECT_ROOT, "assets", "canvas_source.mp4")

# Gaussian-blur radius (pixels) applied to the quad mask's edges so the
# warped content blends smoothly into the camera feed instead of showing a
# hard, aliased polygon edge.
WARP_EDGE_FEATHER_PX = 6

# Whether the warp is active on startup (toggle with 'w').
WARP_ENABLED_BY_DEFAULT = True
