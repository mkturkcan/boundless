"""Smallest end-to-end example: connect, load a map, place an infrastructure camera, capture.

Start the simulator first, e.g.
    Boundless.exe -BoundlessMode=api -BoundlessMap=/Game/Map/Small_City_LVL -windowed -ResX=1920 -ResY=1080
then run:
    python examples/quickstart.py --map "Small City"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boundless import Client, BoundlessError  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default="Small City")
    args = parser.parse_args()

    with Client(args.host, args.port) as client:
        status = client.status()
        print("connected:", status["map"] or "(main menu)", "| mode:", status["mode"])

        if not status["world_ready"] or args.map.lower() not in (status["map"].lower(), status["map_path"].lower()):
            print("loading", args.map, "...")
            client.load_map(args.map)

        # A clear midday scene to start with.
        client.set_lighting(sun_elevation=50, sun_azimuth=140, sun_intensity_scale=4.0, temperature=6000, auto_exposure=True)
        client.set_weather(rain=0, snow=0, snow_cover=0, fog=0)

        for pose in client.sample_camera_poses(n=10, seed=7, kinds=["intersection"]):
            try:
                shot = client.capture(pose=pose, validate_pose=True, save_depth=True)
            except BoundlessError as error:
                print("pose rejected:", error)
                continue
            labels = {}
            for obj in shot.objects:
                labels[obj["label"]] = labels.get(obj["label"], 0) + 1
            print(f"frame {shot.frame}: {shot.image_path}")
            print("  camera:", shot.pose)
            print("  objects:", labels)
            print("  parked vehicles:", sum(1 for o in shot.objects if o.get("parked")))
            break

        client.release_camera()


if __name__ == "__main__":
    main()
