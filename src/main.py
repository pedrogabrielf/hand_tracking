"""
Handtrack Perspective Canvas -- Step 1 + Step 2 + Step 3

Step 1:
    - Open the webcam.
    - Run MediaPipe Hands in real time.
    - Extract and highlight the 4 key landmarks we'll use to build the
      perspective quadrilateral:
        * Left hand  -> thumb tip, index fingertip
        * Right hand -> thumb tip, index fingertip
    - Overlay FPS so we can confirm we're comfortably hitting >= 30 FPS.

Step 2 (see quad.py):
    - Turn those 4 points into a stable, consistently-ordered, smoothed
      quadrilateral and draw it (outline + translucent fill + corner
      labels) live as the hands move.

Step 3 (see warp.py):
    - Warp a source image/video into that quadrilateral with
      cv2.getPerspectiveTransform + cv2.warpPerspective, composited with
      a feathered edge so it blends into the camera feed.

Run with:
    python src/main.py

Controls:
    q / ESC -> quit
    w       -> toggle the perspective warp on/off
"""

import time

import cv2

import camera_utils
import config
from hand_tracker import HandTracker
from quad import QuadrilateralTracker, draw_quadrilateral, is_valid_quad
from warp import PerspectiveWarper, create_source


def resolve_camera_index() -> int:
    """
    Picks the camera index to open. On macOS this prefers a name-based
    lookup (robust against Continuity Camera reordering); everywhere else
    -- or if that lookup fails -- it falls back to config.CAMERA_INDEX.
    """
    resolved = camera_utils.find_camera_index(
        config.PREFERRED_CAMERA_NAME_SUBSTRINGS,
        config.EXCLUDE_CAMERA_NAME_SUBSTRINGS,
    )
    if resolved is not None:
        print(f"Resolved camera by name -> index {resolved}")
        return resolved

    print(
        f"Could not resolve camera by name, falling back to "
        f"config.CAMERA_INDEX = {config.CAMERA_INDEX}"
    )
    return config.CAMERA_INDEX


def open_camera() -> cv2.VideoCapture:
    camera_index = resolve_camera_index()
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera at index {camera_index}. "
            "Check config.CAMERA_INDEX and OS camera permissions."
        )
    return cap


def draw_key_points(frame, key_points) -> None:
    """Draws big, labeled circles on the 4 tracked fingertips."""
    for name, point in key_points.as_dict().items():
        if point is None:
            continue
        color = config.KEY_POINT_COLORS[name]
        cv2.circle(frame, point, config.KEY_POINT_RADIUS, color, -1)
        cv2.putText(
            frame,
            name.replace("_", " "),
            (point[0] + 12, point[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_fps(frame, fps: float) -> None:
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0) if fps >= config.TARGET_FPS else (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def draw_status(frame, key_points, quad_corners, warp_enabled: bool) -> None:
    if not key_points.has_all_points():
        text, color = "Waiting for both hands...", (0, 165, 255)
    elif quad_corners is None:
        text, color = "Hands too close together (quad too small)", (0, 165, 255)
    elif warp_enabled:
        text, color = "READY: warp active  [w] toggle warp", (0, 255, 0)
    else:
        text, color = "READY: quadrilateral active  [w] toggle warp", (0, 255, 0)

    cv2.putText(
        frame, text, (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
    )


def main() -> None:
    cap = open_camera()
    prev_time = time.perf_counter()
    quad_tracker = QuadrilateralTracker()
    warper = PerspectiveWarper()
    source = create_source()
    warp_enabled = config.WARP_ENABLED_BY_DEFAULT

    with HandTracker() as tracker:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Warning: failed to read frame from camera.")
                    continue

                if config.MIRROR_VIEW:
                    frame = cv2.flip(frame, 1)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False  # perf: MediaPipe hint

                key_points, result = tracker.process(frame_rgb)

                if config.SHOW_FULL_SKELETON:
                    tracker.draw_skeleton(frame, result.hand_landmarks)

                quad_corners = quad_tracker.update(key_points)
                if quad_corners is not None and is_valid_quad(quad_corners):
                    if warp_enabled:
                        warper.warp_into(frame, source.get_frame(), quad_corners)
                        draw_quadrilateral(frame, quad_corners, fill_alpha=0)
                    else:
                        draw_quadrilateral(frame, quad_corners)
                else:
                    quad_corners = None  # degenerate quad -> treat as "no quad"

                draw_key_points(frame, key_points)
                draw_status(frame, key_points, quad_corners, warp_enabled)

                now = time.perf_counter()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                draw_fps(frame, fps)

                cv2.imshow(config.WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 'q' or ESC
                    break
                elif key == ord("w"):
                    warp_enabled = not warp_enabled
        finally:
            cap.release()
            source.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
