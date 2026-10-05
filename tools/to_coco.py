"""Convert a Boundless dataset to COCO detection JSON.

    python tools/to_coco.py datasets/run1 --out datasets/run1/coco.json --min-visible 0.1
    python tools/to_coco.py datasets/run1 --visible-box   # use the unoccluded part as the box

Each annotation keeps the 3D information in extra fields (center/extent/rotation in Unreal world
units, corners in OpenCV camera meters) so the same file can drive 2D and 3D experiments.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boundless.labels import filter_objects, iter_frames  # noqa: E402

DEFAULT_CLASSES = ["car", "van", "truck", "bus", "trailer", "pedestrian", "bicycle", "motorcycle"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--out", default=None, help="default: <dataset>/coco.json")
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    parser.add_argument("--min-visible", type=float, default=0.05)
    parser.add_argument("--max-truncation", type=float, default=1.0)
    parser.add_argument("--visible-box", action="store_true", help="use bbox_2d_visible instead of the full projected box")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    category_ids = {name: index + 1 for index, name in enumerate(classes)}

    images, annotations = [], []
    for image_id, (label_path, frame) in enumerate(iter_frames(dataset), start=1):
        if "image" not in frame:
            continue
        camera = frame["camera"]
        images.append({
            "id": image_id,
            "file_name": frame["image"],
            "width": camera["width"],
            "height": camera["height"],
            "map": frame.get("map"),
            "camera_K": camera["K"],
            "camera_world_to_camera": camera["world_to_camera"],
            "pose_kind": camera.get("pose_kind"),
            "scene": frame.get("scene"),
        })
        for obj in filter_objects(frame, min_visible=args.min_visible, max_truncation=args.max_truncation, labels=classes):
            box = obj.get("bbox_2d_visible") if args.visible_box else obj["bbox_2d"]
            if not box:
                continue
            x0, y0, x1, y1 = box
            width, height = x1 - x0, y1 - y0
            if width <= 1 or height <= 1:
                continue
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": category_ids[obj["label"]],
                "bbox": [x0, y0, width, height],
                "area": width * height,
                "iscrowd": 0,
                "track_id": obj["id"],
                "parked": obj.get("parked", False),
                "truncation": obj.get("truncation"),
                "visible_fraction": obj.get("visible_fraction"),
                "occlusion": obj.get("occlusion"),
                "distance_m": obj.get("distance_m"),
                "center_world_cm": obj["center"],
                "extent_cm": obj["extent"],
                "rotation_world_deg": obj["rotation"],
                "corners_camera_m": obj["corners_camera"],
            })

    coco = {
        "info": {"description": f"Boundless export of {dataset.name}"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": n} for n, i in category_ids.items()],
    }
    out = Path(args.out) if args.out else dataset / "coco.json"
    out.write_text(json.dumps(coco), encoding="utf-8")
    print(f"{out}: {len(images)} images, {len(annotations)} boxes")


if __name__ == "__main__":
    main()
