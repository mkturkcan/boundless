"""Summarize a Boundless dataset: frames, cameras, labels, lighting, object statistics.

    python tools/dataset_stats.py datasets/boundless_bigcity_4k [--json stats.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boundless.labels import iter_frames  # noqa: E402


def regime(lighting: dict) -> str:
    scale, elevation = lighting.get("sun_intensity_scale", 1.0), lighting.get("sun_elevation", 45.0)
    if scale < 2.5:
        return "overcast"
    return "low_sun" if elevation < 22 else "sunny"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    frames = 0
    poses = set()
    kinds = Counter()
    labels = Counter()
    sources = Counter()
    parked = Counter()
    regimes = Counter()
    per_frame = []
    visible = []
    occlusion = Counter()
    sizes = Counter()
    for _, frame in iter_frames(args.dataset):
        frames += 1
        camera = frame["camera"]
        poses.add(camera.get("pose_id"))
        kinds[camera.get("pose_kind", "?")] += 1
        # Frames from collect_conditions.py carry the drawn condition; older ones are classified from the lighting.
        condition = frame.get("condition", {}).get("name")
        regimes[condition or regime(frame.get("scene", {}).get("lighting", {}))] += 1
        objects = frame.get("objects", [])
        per_frame.append(len(objects))
        for obj in objects:
            labels[obj["label"]] += 1
            sources[obj["source"]] += 1
            if obj.get("parked"):
                parked[obj["label"]] += 1
            if obj.get("visible_fraction", -1) >= 0:
                visible.append(obj["visible_fraction"])
            occlusion[obj.get("occlusion", 3)] += 1
            x0, y0, x1, y1 = obj["bbox_2d"]
            height = y1 - y0
            sizes["small (<32px)" if height < 32 else "medium (32-96px)" if height < 96 else "large (>=96px)"] += 1

    stats = {
        "frames": frames,
        "camera_poses": len(poses),
        "frames_per_pose": round(frames / max(len(poses), 1), 1),
        "pose_kinds": dict(kinds.most_common()),
        "lighting_regimes": dict(regimes.most_common()),
        "objects_total": sum(per_frame),
        "objects_per_frame_mean": round(statistics.mean(per_frame), 1) if per_frame else 0,
        "objects_per_frame_median": statistics.median(per_frame) if per_frame else 0,
        "frames_without_objects": sum(1 for n in per_frame if n == 0),
        "labels": dict(labels.most_common()),
        "parked_by_label": dict(parked.most_common()),
        "sources": dict(sources.most_common()),
        "occlusion_levels": {str(k): v for k, v in sorted(occlusion.items())},
        "box_heights": dict(sizes.most_common()),
        "visible_fraction_mean": round(statistics.mean(visible), 3) if visible else None,
    }
    print(json.dumps(stats, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
