"""Draw 2D boxes and projected 3D boxes on captured frames (needs Pillow).

    python tools/visualize_labels.py datasets/run1 --out datasets/run1/preview --limit 50
    python tools/visualize_labels.py datasets/run1 --frame 000012 --min-visible 0.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw  # noqa: E402

from boundless.labels import BOX_EDGES, filter_objects, image_path, iter_frames, load_frame  # noqa: E402

COLORS = {
    "car": (66, 165, 245),
    "van": (38, 198, 218),
    "truck": (255, 167, 38),
    "bus": (255, 112, 67),
    "trailer": (141, 110, 99),
    "pedestrian": (102, 187, 106),
    "bicycle": (171, 71, 188),
    "motorcycle": (236, 64, 122),
}


def draw(dataset: Path, label_path: Path, frame: dict, out_dir: Path, min_visible: float, boxes3d: bool) -> Path | None:
    img_path = image_path(dataset, frame)
    if img_path is None or not img_path.exists():
        return None
    image = Image.open(img_path).convert("RGB")
    canvas = ImageDraw.Draw(image)
    for obj in filter_objects(frame, min_visible=min_visible):
        color = COLORS.get(obj["label"], (240, 240, 240))
        if obj.get("parked"):
            color = tuple(int(c * 0.6) for c in color)
        if boxes3d:
            corners = obj["corners_image"]
            for a, b in BOX_EDGES:
                if corners[a] is not None and corners[b] is not None:
                    canvas.line([tuple(corners[a]), tuple(corners[b])], fill=color, width=1)
        x0, y0, x1, y1 = obj.get("bbox_2d_visible", obj["bbox_2d"])
        canvas.rectangle([x0, y0, x1, y1], outline=color, width=2)
        tag = f"{obj['label']}{' P' if obj.get('parked') else ''} {obj.get('visible_fraction', -1):.2f}"
        canvas.text((x0 + 2, max(0, y0 - 11)), tag, fill=color)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{label_path.stem}.jpg"
    image.save(out_path, quality=90)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--out", default=None, help="default: <dataset>/preview")
    parser.add_argument("--frame", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-visible", type=float, default=0.0)
    parser.add_argument("--no-3d", action="store_true")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    out_dir = Path(args.out) if args.out else dataset / "preview"
    if args.frame:
        label_path = dataset / "labels" / f"{args.frame}.json"
        frames = [(label_path, load_frame(label_path))]
    else:
        frames = iter_frames(dataset)

    count = 0
    for label_path, frame in frames:
        written = draw(dataset, label_path, frame, out_dir, args.min_visible, not args.no_3d)
        if written:
            count += 1
            print(written)
        if args.limit and count >= args.limit:
            break


if __name__ == "__main__":
    main()
