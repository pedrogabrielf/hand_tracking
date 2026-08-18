"""
macOS camera resolution helper.

Continuity Camera can register a nearby iPhone as a video device over
AVFoundation, and its position in the device list (and therefore its
OpenCV index) is *not stable* -- it can shift between runs depending on
whether the iPhone is awake, connected, or already streaming to another
app. Hardcoding `CAMERA_INDEX` in config.py is fragile because of this.

This module queries AVFoundation directly (via pyobjc) for the *current*
list of capture devices, in the same order OpenCV's AVFOUNDATION backend
enumerates them, and picks the Mac's own camera by **hardware device
type** (`AVCaptureDeviceTypeBuiltIn*`) rather than by localized name.
Matching by name is unreliable: it depends on OS language (e.g. "Câmera
FaceTime HD" in pt-BR vs "FaceTime HD Camera" in en-US) and the iPhone's
device name is just the phone's own name (e.g. "iPhone de Pedro Gabriel
Camera"), which never contains the word "Continuity". Device type,
however, is a fixed, locale-independent hardware category.

Only works on macOS with `pyobjc-framework-AVFoundation` installed. On any
other platform (or if the lookup fails for any reason) `find_camera_index`
returns None so the caller can fall back to `config.CAMERA_INDEX`.
"""

import platform
from dataclasses import dataclass
from typing import List, Optional, Sequence

# Device types that are physically built into the Mac (never Continuity
# Camera, external webcams, or the iPhone Desk View camera).
_BUILT_IN_DEVICE_TYPE_NAMES = (
    "AVCaptureDeviceTypeBuiltInWideAngleCamera",
    "AVCaptureDeviceTypeBuiltInUltraWideCamera",
    "AVCaptureDeviceTypeBuiltInTelephotoCamera",
    "AVCaptureDeviceTypeBuiltInDualCamera",
    "AVCaptureDeviceTypeBuiltInDualWideCamera",
    "AVCaptureDeviceTypeBuiltInTripleCamera",
    "AVCaptureDeviceTypeBuiltInTrueDepthCamera",
    "AVCaptureDeviceTypeBuiltInLiDARDepthCamera",
)

# Device types we always want to skip, even if somehow name-matched.
_NON_BUILT_IN_DEVICE_TYPE_NAMES = (
    "AVCaptureDeviceTypeContinuityCamera",
    "AVCaptureDeviceTypeExternal",
    "AVCaptureDeviceTypeExternalUnknown",
    "AVCaptureDeviceTypeDeskViewCamera",
)


@dataclass
class _Device:
    name: str
    unique_id: str
    device_type: str
    is_built_in: bool


def _list_capture_devices() -> Optional[List[_Device]]:
    """Returns capture devices in AVFoundation's enumeration order (same
    order OpenCV's AVFOUNDATION backend uses), or None if unavailable
    (non-macOS / pyobjc not installed)."""
    if platform.system() != "Darwin":
        return None

    try:
        import AVFoundation as AVF
    except ImportError:
        return None

    all_type_names = _BUILT_IN_DEVICE_TYPE_NAMES + _NON_BUILT_IN_DEVICE_TYPE_NAMES
    device_types = [
        getattr(AVF, name) for name in all_type_names if hasattr(AVF, name)
    ]

    session = AVF.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
        device_types, AVF.AVMediaTypeVideo, AVF.AVCaptureDevicePositionUnspecified
    )

    result = []
    for d in session.devices():
        device_type = str(d.deviceType())
        result.append(
            _Device(
                name=d.localizedName(),
                unique_id=d.uniqueID(),
                device_type=device_type,
                is_built_in=device_type in _BUILT_IN_DEVICE_TYPE_NAMES,
            )
        )
    return result


def find_camera_index(
    preferred_name_substrings: Sequence[str] = (),
    exclude_name_substrings: Sequence[str] = (),
) -> Optional[int]:
    """
    Finds the OpenCV camera index for the Mac's own built-in camera.

    Primary strategy: match by hardware device type (`AVCaptureDeviceType
    BuiltIn*`), which is reliable regardless of OS language or the
    iPhone's device name. `preferred_name_substrings` /
    `exclude_name_substrings` are only used as a secondary tie-breaker
    (e.g. if a Mac has more than one built-in camera) or as a last-resort
    fallback if no built-in device type is found at all.

    Returns None if unavailable (non-macOS / pyobjc missing) or if no
    devices are found -- callers should fall back to a manually
    configured index in that case.
    """
    devices = _list_capture_devices()
    if not devices:
        return None

    preferred_lower = [s.lower() for s in preferred_name_substrings]
    exclude_lower = [s.lower() for s in exclude_name_substrings]

    def is_excluded(device: _Device) -> bool:
        if device.device_type in _NON_BUILT_IN_DEVICE_TYPE_NAMES:
            return True
        name_lower = device.name.lower()
        return any(bad in name_lower for bad in exclude_lower)

    # 1) Best case: a built-in device type that isn't explicitly excluded.
    for index, device in enumerate(devices):
        if device.is_built_in and not is_excluded(device):
            return index

    # 2) Fallback: no built-in device type found (unlikely on a real Mac) --
    #    try matching by name among the non-excluded devices.
    if preferred_lower:
        for index, device in enumerate(devices):
            if is_excluded(device):
                continue
            if any(good in device.name.lower() for good in preferred_lower):
                return index

    # 3) Last resort: first device that isn't explicitly excluded.
    for index, device in enumerate(devices):
        if not is_excluded(device):
            return index

    return None


def describe_cameras() -> List[str]:
    """Human-readable "index -> name (type)" list, for debugging/logging."""
    devices = _list_capture_devices()
    if not devices:
        return []
    return [
        f"{i} -> {d.name} ({d.device_type}, built_in={d.is_built_in})"
        for i, d in enumerate(devices)
    ]
