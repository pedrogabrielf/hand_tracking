"""
Camera picker / diagnostic tool.

On macOS, "Continuity Camera" can register your iPhone as a video capture
device over AVFoundation. Depending on connection order, it can end up at
index 0 instead of the Mac's built-in FaceTime HD camera -- which is why
`cv2.VideoCapture(0)` sometimes opens the iPhone instead of the webcam.

This script opens every camera index it can find (0..MAX_INDEX) and shows
a live preview, one at a time, so you can visually confirm which index is
your Mac's built-in camera.

Run with:
    python src/list_cameras.py

Controls:
    n / SPACE  -> try next camera index
    s / ENTER  -> select the current index (prints the config.py line to use)
    q / ESC    -> quit without selecting
"""

import cv2

import camera_utils

MAX_INDEX_TO_TRY = 6


def try_open(index: int):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def main() -> None:
    device_names = camera_utils.describe_cameras()
    if device_names:
        print("AVFoundation device names (macOS), in OpenCV index order:")
        for line in device_names:
            print(" ", line)
        print()

    print("Looking for available cameras (indices 0..%d)...\n" % (MAX_INDEX_TO_TRY - 1))

    index = 0
    cap = None

    while index < MAX_INDEX_TO_TRY:
        if cap is None:
            cap = try_open(index)
            if cap is None:
                print(f"[index {index}] could not be opened, skipping.")
                index += 1
                continue

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[index {index}] opened OK ({width}x{height}). Showing preview...")

        ok, frame = cap.read()
        if not ok:
            print(f"[index {index}] opened but failed to read a frame, skipping.")
            cap.release()
            cap = None
            index += 1
            continue

        name_suffix = f" ({device_names[index].split('-> ')[1]})" if index < len(device_names) else ""
        cv2.putText(
            frame,
            f"Camera index: {index}{name_suffix}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "n/SPACE = next   s/ENTER = select   q/ESC = quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Camera Picker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # quit
            break
        elif key in (ord("n"), 32):  # next
            cap.release()
            cap = None
            index += 1
        elif key in (ord("s"), 13):  # select
            print("\nSelected camera index:", index)
            print(f"\nUpdate src/config.py:\n    CAMERA_INDEX = {index}\n")
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
