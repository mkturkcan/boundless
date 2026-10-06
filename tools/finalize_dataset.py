"""Finish a collected dataset: consistency check, stats.json, coco.json, preview.jpg and a README.

    python tools/finalize_dataset.py datasets/boundless_bigcity_4k --title "Boundless Big City 4K"
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

TOOLS = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--title", default="Boundless dataset")
    parser.add_argument("--min-visible", type=float, default=0.1, help="COCO export threshold")
    args = parser.parse_args()
    ds = Path(args.dataset)

    rgb = sorted(p.stem for p in (ds / "rgb").glob("*.jpg"))
    labels = sorted(p.stem for p in (ds / "labels").glob("*.json"))
    manifest = [json.loads(line) for line in (ds / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    problems = []
    if rgb != labels:
        problems.append(f"rgb/labels mismatch ({len(rgb)} vs {len(labels)})")
    if rgb != [f"{i:06d}" for i in range(len(rgb))]:
        problems.append("frame numbering has gaps (run the collector once more, it compacts on start)")
    if len(manifest) != len(rgb):
        problems.append(f"manifest has {len(manifest)} lines for {len(rgb)} frames")
    print("frames:", len(rgb), "; ".join(problems) or "consistent")

    stats = json.loads(subprocess.run([sys.executable, str(TOOLS / "dataset_stats.py"), str(ds), "--json", str(ds / "stats.json")],
                                      capture_output=True, text=True, check=True).stdout)
    subprocess.run([sys.executable, str(TOOLS / "to_coco.py"), str(ds), "--min-visible", str(args.min_visible)], check=True)

    random.seed(0)
    picks = sorted(random.sample(range(len(rgb)), min(16, len(rgb))))
    sheet = Image.new("RGB", (4 * 480, 4 * 270))
    first = None
    for k, i in enumerate(picks):
        frame = f"{i:06d}"
        label = json.loads((ds / "labels" / f"{frame}.json").read_text(encoding="utf-8"))
        first = first or label
        image = Image.open(ds / "rgb" / f"{frame}.jpg").convert("RGB")
        draw = ImageDraw.Draw(image)
        width = max(2, image.width // 640)
        for o in label["objects"]:
            if o.get("visible_fraction", 1) < args.min_visible:
                continue
            color = (110, 220, 110) if o["label"] == "pedestrian" else ((255, 80, 80) if o.get("parked") else (80, 170, 255))
            draw.rectangle(o.get("bbox_2d_visible", o["bbox_2d"]), outline=color, width=width)
        tile = image.resize((480, 270))
        condition = label.get("condition", {}).get("name", "")
        ImageDraw.Draw(tile).text((5, 5), f"{frame} {label['camera']['pose_kind']} {condition} n={len(label['objects'])}", fill=(255, 255, 0))
        sheet.paste(tile, ((k % 4) * 480, (k // 4) * 270))
    sheet.save(ds / "preview.jpg", quality=88)

    size_gb = sum(p.stat().st_size for p in ds.rglob("*") if p.is_file()) / 1e9
    camera = first["camera"] if first else {}
    rows = [("Frames", stats["frames"]),
            ("Camera poses", f"{stats['camera_poses']} (about {stats['frames_per_pose']} frames each)"),
            ("Pose kinds", ", ".join(f"{k}: {v}" for k, v in stats["pose_kinds"].items())),
            ("Conditions", ", ".join(f"{k}: {v}" for k, v in stats["lighting_regimes"].items())),
            ("Objects", f"{stats['objects_total']} ({stats['objects_per_frame_mean']} per frame, median {stats['objects_per_frame_median']})"),
            ("Classes", ", ".join(f"{k}: {v}" for k, v in stats["labels"].items())),
            ("Parked", ", ".join(f"{k}: {v}" for k, v in stats["parked_by_label"].items())),
            ("Sources", ", ".join(f"{k}: {v}" for k, v in stats["sources"].items())),
            ("Box heights", ", ".join(f"{k}: {v}" for k, v in stats["box_heights"].items())),
            ("Occlusion (0/1/2)", ", ".join(f"{k}: {v}" for k, v in stats["occlusion_levels"].items()))]
    readme = f"""# {args.title}

{stats['frames']} frames generated with Boundless (https://github.com/mkturkcan/boundless) on the
{first.get('map', '?') if first else '?'} map. {size_gb:.1f} GB.

- Images: `rgb/NNNNNN.jpg`, {camera.get('width', '?')}x{camera.get('height', '?')} JPEG, game viewport without UI.
- Labels: `labels/NNNNNN.json` (format: https://github.com/mkturkcan/boundless#annotations). Per object: class, source, asset, parked
  flag, 3D box, 8 corners in world / OpenCV camera / pixels, 2D boxes (full, clipped, visible part), truncation,
  depth-tested visible fraction, occlusion level. Per frame: intrinsics K, world-to-camera extrinsics, camera pose,
  lighting/weather/density, and the drawn `condition`.
- `manifest.jsonl`: one line per frame. `coco.json`: COCO export (objects >= {args.min_visible:.0%} visible).
- `stats.json`: the numbers below. `preview.jpg`: 16 random frames with boxes (red = parked).

| | |
|---|---|
""" + "\n".join(f"| {k} | {v} |" for k, v in rows) + """

Frames of one camera are correlated (same viewpoint, different conditions and traffic): split by `camera.pose_id`.
"""
    (ds / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(stats, indent=1)[:2000])


if __name__ == "__main__":
    main()
