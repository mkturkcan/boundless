"""Reading Boundless datasets and small geometry helpers.

A dataset folder looks like:
    session.json          settings used for the run
    manifest.jsonl        one line per frame: frame, image, labels, depth, pose
    rgb/<frame>.png       images (or .jpg)
    labels/<frame>.json   camera + objects (see README)
    depth/<frame>.npy     optional float32 depth in meters (inf = sky)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple

# Corner order of every box (box-local axes: +X forward, +Y right, +Z up):
#   0 (-,-,-) 1 (+,-,-) 2 (+,+,-) 3 (-,+,-)   bottom face
#   4 (-,-,+) 5 (+,-,+) 6 (+,+,+) 7 (-,+,+)   top face
BOX_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def load_frame(label_path: str | Path) -> Dict[str, Any]:
    with open(label_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_frames(dataset_dir: str | Path) -> Iterator[Tuple[Path, Dict[str, Any]]]:
    """Yields (label_path, label_json) for every frame of a dataset folder, in frame order."""
    for path in sorted(Path(dataset_dir, "labels").glob("*.json")):
        yield path, load_frame(path)


def image_path(dataset_dir: str | Path, frame: Dict[str, Any]) -> Path | None:
    rel = frame.get("image")
    return Path(dataset_dir, rel) if rel else None


# -- quaternions (x, y, z, w), Hamilton product as in Unreal's FQuat ---------------------------

def quat_mul(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_conj(q: Sequence[float]) -> Tuple[float, float, float, float]:
    return (-q[0], -q[1], -q[2], q[3])


def rotator_to_quat(pitch: float, yaw: float, roll: float) -> Tuple[float, float, float, float]:
    """Unreal FRotator -> FQuat (same formula as FRotator::Quaternion)."""
    half = math.pi / 360.0
    sp, cp = math.sin(pitch * half), math.cos(pitch * half)
    sy, cy = math.sin(yaw * half), math.cos(yaw * half)
    sr, cr = math.sin(roll * half), math.cos(roll * half)
    return (
        cr * sp * sy - sr * cp * cy,
        -cr * sp * cy - sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def world_to_camera(frame: Dict[str, Any], point_cm: Sequence[float]) -> Tuple[float, float, float]:
    """Unreal world point (cm) -> OpenCV camera coordinates (m) with the frame's extrinsics."""
    m = frame["camera"]["world_to_camera"]
    x, y, z = (v / 100.0 for v in point_cm)
    return tuple(m[r][0] * x + m[r][1] * y + m[r][2] * z + m[r][3] for r in range(3))  # type: ignore[return-value]


def project(frame: Dict[str, Any], point_cm: Sequence[float]) -> Tuple[float, float] | None:
    """Unreal world point (cm) -> pixel (u, v), or None if behind the camera."""
    xc, yc, zc = world_to_camera(frame, point_cm)
    if zc <= 1e-3:
        return None
    k = frame["camera"]["K"]
    return (k[0][0] * xc / zc + k[0][2], k[1][1] * yc / zc + k[1][2])


def filter_objects(frame: Dict[str, Any], min_visible: float = 0.0, max_truncation: float = 1.0,
                   labels: Sequence[str] | None = None) -> List[Dict[str, Any]]:
    out = []
    for obj in frame.get("objects", []):
        visible = obj.get("visible_fraction", -1)
        if visible >= 0 and visible < min_visible:
            continue
        if obj.get("truncation", 0.0) > max_truncation:
            continue
        if labels and obj["label"] not in labels:
            continue
        out.append(obj)
    return out
