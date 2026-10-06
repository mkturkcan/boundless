"""Convert a Boundless dataset to Ultralytics YOLO detection labels.

    python tools/to_yolo.py datasets/run1 --min-visible 0.1 --val-fraction 0.1

Writes <dataset>/yolo/{images,labels}/{train,val}/ (images are hard-linked when possible, copied
otherwise) and <dataset>/yolo/data.yaml. Splits are by camera pose, so frames of one camera never
end up in both train and val.
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boundless.labels import filter_objects, iter_frames  # noqa: E402

DEFAULT_CLASSES = ["car", "van", "truck", "bus", "trailer", "pedestrian", "bicycle", "motorcycle"]


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--out", default=None, help="default: <dataset>/yolo")
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    parser.add_argument("--min-visible", type=float, default=0.1)
    parser.add_argument("--max-truncation", type=float, default=0.7)
    parser.add_argument("--visible-box", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    out = Path(args.out) if args.out else dataset / "yolo"
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    class_ids = {name: index for index, name in enumerate(classes)}

    frames = [(path, frame) for path, frame in iter_frames(dataset) if "image" in frame]
    poses = sorted({frame["camera"].get("pose_id") or path.stem for path, frame in frames})
    random.Random(args.seed).shuffle(poses)
    val_poses = set(poses[: int(round(len(poses) * args.val_fraction))])

    counts = {"train": 0, "val": 0}
    for path, frame in frames:
        split = "val" if (frame["camera"].get("pose_id") or path.stem) in val_poses else "train"
        width, height = frame["camera"]["width"], frame["camera"]["height"]
        lines = []
        for obj in filter_objects(frame, min_visible=args.min_visible, max_truncation=args.max_truncation, labels=classes):
            box = obj.get("bbox_2d_visible") if args.visible_box else obj["bbox_2d"]
            if not box or obj.get("group"):  # group boxes (several objects in one box) have no YOLO equivalent
                continue
            x0, y0, x1, y1 = box
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            cx, cy = (x0 + x1) / 2 / width, (y0 + y1) / 2 / height
            lines.append(f"{class_ids[obj['label']]} {cx:.6f} {cy:.6f} {(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}")

        image_src = dataset / frame["image"]
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        link_or_copy(image_src, out / "images" / split / image_src.name)
        (out / "labels" / split / f"{image_src.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        counts[split] += 1

    names = "\n".join(f"  {i}: {n}" for n, i in class_ids.items())
    (out / "data.yaml").write_text(f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n", encoding="utf-8")
    print(f"{out}: train={counts['train']} val={counts['val']} images, {len(classes)} classes")


if __name__ == "__main__":
    main()
