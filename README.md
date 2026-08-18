# Handtrack Perspective Canvas

Real-time augmented-reality canvas controlled with your bare hands. Four
fingertip points (left thumb, left index, right thumb, right index) define a
dynamic quadrilateral in mid-air, and images/video/filters get warped into
that quadrilateral live via OpenCV's perspective transform.

This repo is being built incrementally:

- **Step 1 (done):** webcam capture + real-time hand landmark detection,
  extracting and visualizing the 4 key fingertip points.
- **Step 2 (done):** compute and draw the dynamic quadrilateral from
  the 4 points.
- **Step 3 (this step):** perspective warp (`cv2.getPerspectiveTransform` +
  `cv2.warpPerspective`) to project an image/video inside the quadrilateral.
- **Step 4:** interactive filters/effects applied within the quadrilateral.

## Requirements

- Python 3.10+ (tested on 3.13)
- A webcam
- macOS / Windows / Linux

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Download the hand landmark model

MediaPipe 1.0 moved hand tracking to the new **Tasks API**, which requires a
model bundle that is not shipped inside the pip package. Download it once:

```bash
mkdir -p models
curl -sSL -o models/hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
```

(`HandTracker` will raise a clear `FileNotFoundError` with this same command
if the file is missing.)

## Run (Step 1 + Step 2 + Step 3)

```bash
python src/main.py
```

You should see your webcam feed, mirrored, with:

- The full 21-point hand skeleton for every detected hand (toggle via
  `config.SHOW_FULL_SKELETON`).
- Four highlighted, labeled circles for `left_thumb`, `left_index`,
  `right_thumb`, `right_index` whenever those points are detected.
- **A dynamic quadrilateral** connecting the 4 points, which now shows the
  warped source image (`assets/canvas_source.jpg` by default) live inside
  it, with feathered edges so it blends into the camera feed (see "The
  perspective warp" below).
- An FPS counter (green when >= 30 FPS, red otherwise).
- A status line: waiting for both hands -> hands too close together (quad
  rejected as degenerate) -> `READY: warp active`.

Controls:

- **q** / **ESC** -> quit
- **w** -> toggle the warp on/off (when off, you see Step 2's plain
  outline + translucent fill instead)

### The quadrilateral (`quad.py`)

Corner order is fixed as **top-left = left index, top-right = right
index, bottom-right = right thumb, bottom-left = left thumb** --
matching the natural "framing" gesture where both index fingers point up
and both thumbs point down, like a photographer framing a shot with their
hands. This fixed order is what Step 3 will rely on to map the warped
image's corners consistently.

`QuadrilateralTracker` also:

- **Smooths** corner positions frame-to-frame with an exponential moving
  average (`config.QUAD_SMOOTHING_ALPHA`) -- raw landmarks are noisy, and
  a trembling quad outline looks bad.
- **Resets** its smoothing state whenever any of the 4 points isn't
  detected, so the quad doesn't "snap" from a stale position once both
  hands reappear.
- **Rejects degenerate quads** (`config.QUAD_MIN_AREA`) -- if your hands
  get too close together the points become nearly collinear, which would
  otherwise break `cv2.getPerspectiveTransform` in Step 3.

### The perspective warp (`warp.py`)

`PerspectiveWarper.warp_into()` maps the source content's own 4 corners
onto the quad's 4 corners with `cv2.getPerspectiveTransform`, warps it
with `cv2.warpPerspective`, and composites it onto the camera frame with
a Gaussian-feathered alpha mask (`config.WARP_EDGE_FEATHER_PX`) so the
edge doesn't look like a hard-edged polygon cutout.

Performance: the warp + composite only touches the quad's bounding-box
region of the frame (plus a small margin for feathering), not the full
1280x720 image -- processing the whole frame every frame regardless of
how small the hand-held quad is would be wasteful and hurts the >=30 FPS
target.

Content source (`config.WARP_SOURCE_TYPE`):

- `"image"` (default) -- a static image, `config.WARP_SOURCE_IMAGE_PATH`
  (defaults to the generated `assets/canvas_source.jpg`). Swap in your
  own image by changing that path.
- `"video"` -- a looping video file, `config.WARP_SOURCE_VIDEO_PATH`.

Both are implemented in `warp.py` as tiny `ImageSource` / `VideoSource`
classes with a `get_frame()` method, so Step 4's filters can wrap around
either one the same way.

### Wrong camera opening on macOS (Continuity Camera)

If the webcam feed shows your iPhone's camera instead of the Mac's built-in
one, that's **Continuity Camera**: macOS can register a nearby iPhone as a
video device. Its position in the device list is **not stable** across
runs -- it can shift ahead of or behind the built-in camera depending on
whether the iPhone is awake/connected -- so a hardcoded index alone isn't
reliable.

To handle this, `main.py` resolves the camera at startup via
`camera_utils.find_camera_index` (using `pyobjc-framework-AVFoundation`),
which matches by **hardware device type**
(`AVCaptureDeviceTypeBuiltIn*`) -- not by localized name. Name matching
alone is unreliable: the built-in camera's name changes with the OS
language (e.g. `"Câmera FaceTime HD"` in pt-BR vs `"FaceTime HD Camera"`
in en-US), and the iPhone's device name is just the phone's own name
(e.g. `"iPhone de Pedro Gabriel Camera"`), which never actually contains
the word `"Continuity"`. `config.PREFERRED_CAMERA_NAME_SUBSTRINGS` /
`EXCLUDE_CAMERA_NAME_SUBSTRINGS` are only used as a secondary
tie-breaker/fallback. It prints which index it resolved to on startup,
e.g. `Resolved camera by name -> index 1`.

If that lookup fails entirely (non-macOS, `pyobjc` not installed, or no
devices found), it falls back to `config.CAMERA_INDEX`. In that case:

1. Run the picker tool to find the right index visually:

   ```bash
   python src/list_cameras.py
   ```

   It also prints the AVFoundation device names in index order. Press
   `n`/`SPACE` to cycle through camera indices, and `s`/`ENTER` once you
   see the Mac's own camera feed. It prints the `CAMERA_INDEX` value to
   set in `src/config.py`.

2. Or disable Continuity Camera on the iPhone: **Settings > General >
   AirPlay & Handoff > Continuity Camera** (turn off), or simply keep the
   iPhone locked / out of Bluetooth range while running the app.

## Project layout

```
hand_tracking/
├── requirements.txt
├── models/
│   └── hand_landmarker.task     # downloaded, not committed
├── assets/
│   └── canvas_source.jpg        # default Step 3 warp source image
└── src/
    ├── config.py                # all tunables in one place
    ├── camera_utils.py          # macOS camera resolution (Continuity Camera fix)
    ├── list_cameras.py          # interactive camera index picker (diagnostic)
    ├── hand_tracker.py          # MediaPipe HandLandmarker wrapper
    ├── quad.py                  # Step 2: quadrilateral tracking + drawing
    ├── warp.py                  # Step 3: perspective warp + content sources
    └── main.py                  # entry point (Step 1 + Step 2 + Step 3)
```

## Notes on handedness

MediaPipe's handedness classifier assumes a mirrored (selfie-style) input
image. `config.MIRROR_VIEW = True` flips the frame *before* running
detection, so the `Left`/`Right` labels MediaPipe returns match the user's
own left/right hands directly (no manual swapping needed).
