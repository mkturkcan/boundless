"""Unattended, resumable dataset generation with varied viewpoints, time of day, weather and traffic.

The city is visited as a tour of *sites* (an intersection and its surroundings). At every site:

  * the area is loaded once (the next site is preloaded in the background), and up to ``--views-per-site`` distinct
    infrastructure cameras are placed: intersection and mid-block poles, building facades;
  * ``--rounds-per-site`` rounds are taken. Every round draws a new condition (sunny, low sun, overcast, rain, snow,
    dusk, night, night rain; see ``CONDITIONS``), re-spawns moving traffic, parked vehicles and pedestrians at new
    densities, waits for them to settle, and captures every camera once.

A camera view is therefore only ever repeated after all agents have been re-spawned. After a re-spawn the simulation
runs for ``--respawn-wait`` seconds (agents fade in, overlapping vehicles resolve, traffic starts to flow), and after
every camera move for ``--view-settle-seconds`` (nearby agents switch to their full representation). The simulator is launched (and relaunched after a
crash) when ``--launch`` is given; a run can be stopped at any time and resumes where it left off. The drawn condition
is stored in every label file under ``"condition"``.

    python examples/collect_dataset.py --out datasets/boundless_bigcity_4k --target 10000 --launch C:/Boundless/Boundless.exe

    # preview: 2 frames of every condition plus a contact sheet (gallery.jpg)
    python examples/collect_dataset.py --out datasets/preview --gallery 2 --launch C:/Boundless/Boundless.exe
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boundless import BoundlessError, Client, Pose  # noqa: E402

# Camera mix: intersection poles dominate; facade cameras sit at 2nd/3rd-floor windows.
RIG = {"intersection_weight": 0.7, "mid_block_weight": 0.2, "freeway_weight": 0.1, "facade_weight": 0.15,
       "min_floor": 2, "max_floor": 3}

MAPS = {"big city": "/Game/Map/Big_City_LVL", "small city": "/Game/Map/Small_City_LVL"}


def map_path(name: str) -> str:
    return MAPS.get(name.lower(), name)


CONDITIONS = [  # name, probability
    ("sunny", 0.27),
    ("low_sun", 0.10),
    ("overcast", 0.15),
    ("rain", 0.14),
    ("snow", 0.07),
    ("dusk", 0.05),
    ("night", 0.16),
    ("night_rain", 0.06),
]


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


# --------------------------------------------------------------------------------------------------------------------
# Conditions. Lighting scales are relative to the map's authored values for the current mode (day or night).
# Auto exposure normalizes brightness like a real camera; exposure_bias and white_balance add camera variance.
# --------------------------------------------------------------------------------------------------------------------

def U(rng: random.Random, lo: float, hi: float) -> float:
    return rng.uniform(lo, hi)


def log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def tint(rng: random.Random, base, jitter: float):
    return [max(0.0, b + rng.uniform(-jitter, jitter)) for b in base]


def street_azimuth(rng: random.Random, street_yaw: float | None) -> float:
    """Sun azimuth along one of the site's street axes (+-20 deg): in a dense city that is when sunlight reaches the
    street. Without a street direction, any azimuth."""
    if street_yaw is None:
        return U(rng, 0, 360)
    return (street_yaw + rng.choice([0.0, 90.0, 180.0, 270.0]) + U(rng, -20, 20)) % 360.0


def day_lighting(rng: random.Random, street_yaw: float | None = None, **override) -> dict:
    elevation = U(rng, 12, 85)
    lighting = dict(
        night=False, sun_elevation=elevation, sun_azimuth=U(rng, 0, 360),
        temperature=U(rng, 4800, 7800) if elevation > 25 else U(rng, 4000, 6000),
        sun_intensity_scale=U(rng, 2.0, 8.0), sky_intensity_scale=U(rng, 0.6, 1.4),
        sky_dome_intensity=U(rng, 0.7, 1.3), sky_dome_tint=tint(rng, [1, 1, 1], 0.08), sky_sun_intensity=1.0,
        exposure_bias=U(rng, -0.8, 0.5), auto_exposure=True, auto_exposure_range=[-2.0, 16.0],
        saturation=U(rng, 0.85, 1.15), contrast=U(rng, 0.92, 1.1), white_balance=U(rng, 5600, 7400))
    lighting.update(override)
    return lighting


def night_lighting(rng: random.Random, **override) -> dict:
    lighting = dict(
        night=True, sun_elevation=U(rng, 20, 75), sun_azimuth=U(rng, 0, 360), temperature=U(rng, 7500, 12000),
        sun_intensity_scale=U(rng, 1.0, 15.0), sky_intensity_scale=U(rng, 0.5, 4.0),
        sky_dome_intensity=U(rng, 1.0, 8.0), sky_dome_tint=tint(rng, [1, 1, 1], 0.1), sky_sun_intensity=1.0,
        exposure_bias=U(rng, -4.2, -2.4), auto_exposure=True, auto_exposure_range=[-4.0, 16.0],
        saturation=U(rng, 0.85, 1.1), contrast=U(rng, 0.95, 1.15),
        # Camera white balance under sodium street lights: mostly neutral-to-cool, sometimes left warm.
        white_balance=U(rng, 3000, 4800) if rng.random() < 0.75 else U(rng, 5500, 6500))
    lighting.update(override)
    return lighting


def sample_condition(rng: random.Random, name: str | None = None, street_yaw: float | None = None) -> dict:
    if name is None:
        pick = rng.random()
        for name, probability in CONDITIONS:
            pick -= probability
            if pick <= 0:
                break
    weather = dict(rain=0.0, snow=0.0, snow_cover=0.0, fog=0.0)
    if name == "sunny":
        # Sun along a street axis and fairly high, so the street itself is sunlit.
        lighting = day_lighting(rng, sun_elevation=U(rng, 30, 80), sun_azimuth=street_azimuth(rng, street_yaw))
        weather["fog"] = U(rng, 0.0, 0.15)
    elif name == "low_sun":
        lighting = day_lighting(rng, sun_elevation=U(rng, 2, 12), sun_azimuth=street_azimuth(rng, street_yaw), temperature=U(rng, 2800, 4500),
                                sun_intensity_scale=U(rng, 2.0, 7.0), sky_intensity_scale=U(rng, 0.5, 1.0),
                                sky_dome_tint=tint(rng, [1.0, 0.85, 0.7], 0.08))
        weather["fog"] = U(rng, 0.0, 0.3)
    elif name == "overcast":
        lighting = day_lighting(rng, sun_intensity_scale=U(rng, 0.01, 0.2), sky_intensity_scale=U(rng, 0.9, 1.5),
                                sky_dome_intensity=U(rng, 0.45, 0.9), sky_dome_tint=tint(rng, [0.62, 0.66, 0.72], 0.06),
                                sky_sun_intensity=U(rng, 0.0, 0.05), temperature=U(rng, 6000, 7800),
                                saturation=U(rng, 0.75, 0.95))
        weather["fog"] = U(rng, 0.05, 0.5)
    elif name == "rain":
        lighting = day_lighting(rng, sun_intensity_scale=U(rng, 0.0, 0.06), sky_intensity_scale=U(rng, 0.8, 1.3),
                                sky_dome_intensity=U(rng, 0.3, 0.65), sky_dome_tint=tint(rng, [0.52, 0.57, 0.62], 0.05),
                                sky_sun_intensity=0.0, temperature=U(rng, 6200, 8000), exposure_bias=U(rng, -0.7, 0.2),
                                saturation=U(rng, 0.7, 0.9), contrast=U(rng, 0.9, 1.0))
        weather.update(rain=log_uniform(rng, 0.08, 0.6), fog=U(rng, 0.15, 0.7))
    elif name == "snow":
        lighting = day_lighting(rng, sun_intensity_scale=U(rng, 0.0, 0.5), sky_intensity_scale=U(rng, 0.9, 1.4),
                                sky_dome_intensity=U(rng, 0.5, 0.9), sky_dome_tint=tint(rng, [0.68, 0.72, 0.78], 0.05),
                                sky_sun_intensity=U(rng, 0.0, 0.1), temperature=U(rng, 6500, 8500),
                                saturation=U(rng, 0.75, 0.95))
        # Most shots while it snows, some after the snowfall (cover only).
        weather.update(snow_cover=1.0, snow=0.0 if rng.random() < 0.3 else log_uniform(rng, 0.1, 1.0), fog=U(rng, 0.1, 0.6))
    elif name == "dusk":
        # Night setup (street lights, lit windows) under a much brighter, warm-blue sky and a dim low sun.
        lighting = night_lighting(rng, sun_elevation=U(rng, 3, 12), temperature=U(rng, 2500, 3800),
                                  sun_intensity_scale=U(rng, 500, 4000), sky_intensity_scale=U(rng, 30, 150),
                                  sky_dome_intensity=U(rng, 300, 2000), sky_dome_tint=tint(rng, [0.8, 0.75, 0.95], 0.1),
                                  exposure_bias=U(rng, -1.5, -0.3), saturation=U(rng, 0.9, 1.15),
                                  contrast=U(rng, 0.95, 1.1), white_balance=U(rng, 4500, 6500))
        weather["fog"] = U(rng, 0.0, 0.3)
    elif name in ("night", "night_rain"):
        lighting = night_lighting(rng)
        weather["fog"] = U(rng, 0.0, 0.25)
        if name == "night_rain":
            # Street lights make rain streaks far brighter than by day: keep it lighter.
            weather.update(rain=log_uniform(rng, 0.02, 0.12), fog=U(rng, 0.1, 0.5))
    else:
        raise ValueError(f"unknown condition {name!r}")
    return dict(name=name, lighting=lighting, weather=weather)


def apply_condition(client: Client, condition: dict) -> None:
    client.set_lighting(**condition["lighting"])
    client.set_weather(**condition["weather"])


def respawn_traffic(client: Client, rng: random.Random, args) -> dict:
    density = dict(traffic=U(rng, *args.traffic_range), parked=U(rng, *args.parked_range), crowd=U(rng, *args.crowd_range))
    client.set_density(respawn=True, **density)
    return density


def tag_label(shot, condition: dict, density: dict, site: str, round_index: int) -> None:
    """Store the drawn condition in the frame's label file."""
    if not shot.label_path:
        return
    path = Path(shot.label_path)
    label = json.loads(path.read_text(encoding="utf-8"))
    label["condition"] = dict(name=condition["name"], weather=condition["weather"], site=site, round=round_index, density=density)
    path.write_text(json.dumps(label), encoding="utf-8")


# --------------------------------------------------------------------------------------------------------------------
# Simulator process
# --------------------------------------------------------------------------------------------------------------------

class Simulator:
    """Owns at most one simulator process."""

    def __init__(self, args):
        self.args = args
        self.process = None

    def command(self) -> list[str]:
        width, height = self.args.resolution.lower().split("x")
        command = [self.args.launch]
        if self.args.project:  # editor build: UnrealEditor.exe <project> -game
            command += [self.args.project, "-game"]
        return command + ["-windowed", f"-ResX={width}", f"-ResY={height}", "-ForceRes",
                          "-BoundlessMode=api", f"-BoundlessMap={map_path(self.args.map)}", f"-BoundlessPort={self.args.port}",
                          "-nosound", "-unattended", "-nosplash"]

    def start(self) -> None:
        self.stop()
        log("launching: " + " ".join(self.command()))
        self.process = subprocess.Popen(self.command())

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            log("stopping the simulator")
            self.process.terminate()
            try:
                self.process.wait(60)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        # The launcher may leave the game process behind; end whatever still listens on the API port.
        for pid in listening_pids(self.args.port):
            log(f"ending leftover simulator process {pid}")
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
        time.sleep(2)


def listening_pids(port: int) -> set[int]:
    if sys.platform != "win32":
        return set()
    lines = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True).stdout.splitlines()
    return {int(p[-1]) for p in (line.split() for line in lines)
            if len(p) >= 5 and p[1].endswith(f":{port}") and p[3] == "LISTENING" and p[-1] != "0"}


def connect(args, simulator: Simulator | None) -> Client:
    try:
        return Client(args.host, args.port, timeout=args.call_timeout, connect_timeout=10)
    except ConnectionError:
        if simulator is None:
            raise
    simulator.start()
    return Client(args.host, args.port, timeout=args.call_timeout, connect_timeout=1800)


def prepare_session(client: Client, args) -> None:
    status = client.status()
    if not (status.get("world_ready") and status.get("map_path", "").lower() == map_path(args.map).lower()):
        log(f"loading {args.map}")
        client.load_map(args.map, timeout=3600)
    client.wait_until_ready(timeout=3600)
    width, height = args.resolution.lower().split("x")
    # Native-resolution rendering, no motion blur, and the window forced to the requested size.
    for command in ("r.ScreenPercentage.MaxResolution 0", "r.ScreenPercentage 100", "r.MotionBlurQuality 0",
                    f"r.SetRes {width}x{height}w"):
        client.console(command)
    time.sleep(2)


def order_by_route(poses: list[Pose]) -> list[Pose]:
    """Greedy nearest-neighbour tour, so the city streams in gradually between cameras."""
    remaining = list(poses)
    route = [remaining.pop(0)] if remaining else []
    while remaining:
        last = route[-1].location
        best = min(range(len(remaining)), key=lambda i: math.dist(last[:2], remaining[i].location[:2]))
        route.append(remaining.pop(best))
    return route


def build_route(client: Client, args, seed: int) -> list[Pose]:
    candidates = client.sample_camera_poses(n=args.candidates, seed=seed, kinds=args.kinds.split(","), rig=RIG)
    subset = random.Random(seed).sample(candidates, min(args.route_poses, len(candidates)))
    return order_by_route(subset)


def capture_options(args, **override) -> dict:
    options = dict(image_format="jpg", jpeg_quality=args.jpeg_quality, max_distance=args.max_distance_m * 100.0,
                   min_visible_pixels=args.min_visible_pixels, min_visible_fraction=args.min_visible_fraction,
                   depth_downscale=2, save_depth=args.save_depth, freeze_mode="None", return_annotations=False)
    options.update(override)
    return options


# --------------------------------------------------------------------------------------------------------------------
# Dataset bookkeeping
# --------------------------------------------------------------------------------------------------------------------

def count_frames(out: Path) -> int:
    return len(list((out / "rgb").glob("*.jpg"))) if (out / "rgb").exists() else 0


def reject_frame(out: Path, frame: str, pair) -> None:
    """Move a frame to rejected/ (kept for inspection) together with the reason."""
    target = out / "rejected"
    target.mkdir(exist_ok=True)
    for path in (out / "rgb" / f"{frame}.jpg", out / "labels" / f"{frame}.json", out / "depth" / f"{frame}.npy"):
        if path.exists():
            path.replace(target / path.name)
    (target / f"{frame}.reason.json").write_text(json.dumps({"overlap": [pair[0]["id"], pair[1]["id"]]}), encoding="utf-8")


def discard_frame(out: Path, frame: str) -> None:
    for path in (out / "rgb" / f"{frame}.jpg", out / "labels" / f"{frame}.json", out / "depth" / f"{frame}.npy"):
        if path.exists():
            path.unlink()


def compact_dataset(out: Path) -> int:
    """Renumber frames 0..N-1 (closing gaps left by discarded frames) and rewrite the manifest."""
    manifest_path = out / "manifest.jsonl"
    by_frame = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                by_frame[entry["frame"]] = entry
    existing = sorted(p.stem for p in (out / "rgb").glob("*.jpg")) if (out / "rgb").exists() else []
    entries = []
    for index, old in enumerate(existing):
        new = f"{index:06d}"
        if old != new:
            (out / "rgb" / f"{old}.jpg").rename(out / "rgb" / f"{new}.jpg")
            label_path = out / "labels" / f"{old}.json"
            if label_path.exists():
                label = json.loads(label_path.read_text(encoding="utf-8"))
                label["frame"], label["image"] = new, f"rgb/{new}.jpg"
                (out / "labels" / f"{new}.json").write_text(json.dumps(label), encoding="utf-8")
                label_path.unlink()
            depth_path = out / "depth" / f"{old}.npy"
            if depth_path.exists():
                depth_path.rename(out / "depth" / f"{new}.npy")
        entry = by_frame.get(old, {"frame": new})
        entry.update(frame=new, image=f"rgb/{new}.jpg", labels=f"labels/{new}.json")
        entries.append(entry)
    if existing or manifest_path.exists():
        manifest_path.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    return len(existing)


# --------------------------------------------------------------------------------------------------------------------
# Gallery: a few frames of every condition, to check the presets before a long run.
# --------------------------------------------------------------------------------------------------------------------

def run_gallery(client: Client, args) -> None:
    from PIL import Image, ImageDraw

    out = Path(args.out)
    rng = random.Random(args.seed)
    route = build_route(client, args, args.seed)
    tiles = []
    cursor = 0
    wanted = args.gallery_conditions.split(",") if args.gallery_conditions else [name for name, _ in CONDITIONS]
    for name in wanted:
        made = 0
        while made < args.gallery and cursor < len(route):
            pose = route[cursor]
            cursor += 1
            condition = sample_condition(rng, name)
            apply_condition(client, condition)
            density = respawn_traffic(client, rng, args)
            try:
                shot = client.capture(pose=pose, validate_pose=True, output_dir=str(out.resolve()), name=f"{name}_{made}",
                                      **capture_options(args, settle_seconds=args.respawn_wait + 1.0, settle_frames=40))
            except BoundlessError as error:
                log(f"  skip {pose.id}: {str(error).split(': ', 1)[-1]}")
                continue
            tag_label(shot, condition, density, pose.id, 0)
            made += 1
            tiles.append((f"{name}  {shot.raw.get('num_objects', 0)} objects", shot.image_path))
            log(f"{name}: {shot.image_path}")
    client.set_weather(rain=0, snow=0, snow_cover=0, fog=0)
    cols = 4
    sheet = Image.new("RGB", (cols * 640, ((len(tiles) + cols - 1) // cols) * 360))
    for i, (label, path) in enumerate(tiles):
        tile = Image.open(path).convert("RGB").resize((640, 360))
        ImageDraw.Draw(tile).text((6, 6), label, fill=(255, 255, 0))
        sheet.paste(tile, ((i % cols) * 640, (i // cols) * 360))
    sheet.save(out / "gallery.jpg", quality=88)
    log(f"gallery: {out / 'gallery.jpg'}")


# --------------------------------------------------------------------------------------------------------------------

def yaw_difference(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def site_views(client: Client, center, args, seed: int) -> list[Pose]:
    """Distinct, geometry-checked camera poses around a site (different mount points or directions)."""
    candidates = client.sample_camera_poses(n=args.site_candidates, seed=seed, kinds=args.kinds.split(","),
                                            near=center, radius=args.site_view_radius_m * 100.0, rig=RIG)
    views: list[Pose] = []
    for candidate in candidates:
        try:
            checked = client.validate_pose(candidate)
        except BoundlessError:
            continue
        if not checked.get("valid"):
            continue
        pose = Pose.from_dict(checked["pose"])
        if not -45.0 <= pose.rotation[0] <= 5.0:
            continue
        if any(math.dist(pose.location, other.location) < 600.0 and yaw_difference(pose.rotation[1], other.rotation[1]) < 40.0
               for other in views):
            continue
        views.append(pose)
        if len(views) >= args.views_per_site:
            break
    return views


def footprints_overlap(a, b, min_depth_cm: float) -> bool:
    """Separating-axis test on two oriented rectangles (lists of 4 XY corners): True if they interpenetrate by more
    than min_depth_cm along every axis."""
    for rect in (a, b):
        for k in range(4):
            ex, ey = rect[(k + 1) % 4][0] - rect[k][0], rect[(k + 1) % 4][1] - rect[k][1]
            length = math.hypot(ex, ey) or 1.0
            nx, ny = -ey / length, ex / length
            pa = [x * nx + y * ny for x, y in a]
            pb = [x * nx + y * ny for x, y in b]
            if min(max(pa), max(pb)) - max(min(pa), min(pb)) < min_depth_cm:
                return False
    return True


def overlapping_vehicles(label_path: str, min_depth_cm: float = 50.0):
    """The first pair of vehicles within 150 m of the camera that interpenetrate (spawned into each other and still
    resolving), or None. Trailers are skipped: they legitimately touch their truck."""
    if not label_path:
        return None
    objects = json.loads(Path(label_path).read_text(encoding="utf-8")).get("objects", [])
    vehicles = [o for o in objects if o["label"] not in ("pedestrian", "trailer") and o.get("distance_m", 1e9) < 150.0
                and o.get("visible_fraction", 0) > 0.1]
    rects = [[(c[0], c[1]) for c in o["corners_world"][:4]] for o in vehicles]
    heights = [(min(c[2] for c in o["corners_world"]), max(c[2] for c in o["corners_world"])) for o in vehicles]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            # Same level (not a car on an overpass above another) and footprints interpenetrating.
            vertical = min(heights[i][1], heights[j][1]) - max(heights[i][0], heights[j][0])
            if vertical > min_depth_cm and footprints_overlap(rects[i], rects[j], min_depth_cm):
                return vehicles[i], vehicles[j]
    return None


def wait_for_site(client: Client, slot: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if client.site_ready(slot):
            return True
        time.sleep(0.5)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target", type=int, default=10000)
    parser.add_argument("--map", default="Big City", help='"Big City", "Small City" or a map path')
    parser.add_argument("--resolution", default="3840x2160")
    parser.add_argument("--kinds", default="intersection,midblock,facade,freeway")
    parser.add_argument("--candidates", type=int, default=6000, help="intersection poses drawn to pick sites from")
    parser.add_argument("--route-sites", type=int, default=1200, help="city-wide subset of sites visited along a short tour")
    parser.add_argument("--views-per-site", type=int, default=16)
    parser.add_argument("--min-views", type=int, default=3, help="skip sites with fewer usable cameras")
    parser.add_argument("--rounds-per-site", type=int, default=3, help="condition + re-spawn rounds per site")
    parser.add_argument("--site-candidates", type=int, default=120)
    parser.add_argument("--site-view-radius-m", type=float, default=140.0, help="cameras are placed within this distance of the site")
    parser.add_argument("--site-stream-radius-m", type=float, default=200.0, help="area kept loaded around a site")
    parser.add_argument("--site-stream-timeout", type=float, default=240.0)
    parser.add_argument("--respawn-wait", type=float, default=6.0,
                        help="minimum seconds after a re-spawn (vehicles spawned into each other resolve); capturing also waits until "
                             "every agent in view has loaded")
    parser.add_argument("--view-settle-frames", type=int, default=4,
                        help="extra frames rendered before the shot; the readiness wait already holds the camera still for ~1 s")
    parser.add_argument("--settle-timeout", type=float, default=8.0, help="maximum seconds to wait for agents in view to load")
    parser.add_argument("--actor-spawn-budget", type=float, default=0.05,
                        help="seconds per frame the simulator may spend spawning agents (gameplay default 0.0015)")
    parser.add_argument("--traffic-range", type=float, nargs=2, default=[0.4, 1.6])
    parser.add_argument("--parked-range", type=float, nargs=2, default=[0.3, 1.2])
    parser.add_argument("--crowd-range", type=float, nargs=2, default=[0.3, 1.6])
    parser.add_argument("--max-distance-m", type=float, default=2000.0, help="label every rendered object up to this range")
    parser.add_argument("--min-visible-pixels", type=int, default=4, help="at depth-map resolution (half the image size)")
    parser.add_argument("--min-visible-fraction", type=float, default=0.01)
    parser.add_argument("--save-depth", action="store_true", help="also write depth/<frame>.npy (float32 meters)")
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--max-hours", type=float, default=0.0, help="stop after this long (0 = no limit)")
    parser.add_argument("--gallery", type=int, default=0, help="only render N frames per condition into --out")
    parser.add_argument("--gallery-conditions", default="", help="comma-separated subset for --gallery")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--call-timeout", type=float, default=600.0)
    parser.add_argument("--launch", default=None, help="simulator executable; enables automatic (re)launch")
    parser.add_argument("--restart-every-sites", type=int, default=3,
                        help="with --launch: restart the simulator after this many sites to keep frame times steady")
    parser.add_argument("--project", default=None, help="only for editor builds: the .uproject passed to the editor")
    args = parser.parse_args()
    args.route_poses = args.route_sites
    args.segments_per_pose = 1

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    args.out = str(out)
    simulator = Simulator(args) if args.launch else None

    if args.gallery:
        client = connect(args, simulator)
        prepare_session(client, args)
        run_gallery(client, args)
        client.close()
        return

    state_path = out / "collection_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    state = {"site_cursor": 0, "sites_done": 0, "sites_skipped": 0, "route_seed": args.seed, **state}
    rng = random.Random(args.seed + state["sites_done"] * 7919 + state["site_cursor"])
    started = time.time()
    client = None
    sites: list[Pose] = []
    width, height = (int(v) for v in args.resolution.lower().split("x"))
    frames = count_frames(out)
    frames_at_start = frames
    stream_radius = args.site_stream_radius_m * 100.0
    # After every camera move the collector waits until no agent in view is still loading in (client.readiness), then
    # renders a few more frames for lighting and anti-aliasing history.
    view_options = capture_options(args, settle_seconds=0.0, settle_frames=args.view_settle_frames,
                                   freeze_frames=0, streaming_timeout_seconds=5.0)
    size_checked = False
    sites_since_launch = 0

    def anchors(seed: int) -> list[Pose]:
        candidates = client.sample_camera_poses(n=args.candidates, seed=seed, kinds=["intersection"], rig=RIG)
        subset = random.Random(seed).sample(candidates, min(args.route_sites, len(candidates)))
        return order_by_route(subset)

    while frames < args.target:
        if args.max_hours and time.time() - started > args.max_hours * 3600:
            log(f"time limit reached: {frames} frames")
            break
        try:
            if client is None:
                frames = compact_dataset(out)
                client = connect(args, simulator)
                prepare_session(client, args)
                client.set_actor_spawn_budget(args.actor_spawn_budget)
                size_checked = False
                sites = anchors(state["route_seed"])
                log(f"{len(sites)} sites on the route, resuming at {state['site_cursor']}")
                sites_since_launch = 0
            elif simulator is not None and args.restart_every_sites and sites_since_launch >= args.restart_every_sites:
                log(f"restarting the simulator after {sites_since_launch} sites")
                client.close()
                client = None
                simulator.stop()
                continue
            if state["site_cursor"] >= len(sites):
                state["site_cursor"] = 0
                state["route_seed"] += 1
                sites = anchors(state["route_seed"])
            index = state["site_cursor"]
            slot = index % 2
            anchor = sites[index]
            t0 = time.time()

            # Load this site and preload the next one while it is being captured.
            client.stream_site(anchor.location, stream_radius, slot=slot)
            if index + 1 < len(sites):
                client.stream_site(sites[index + 1].location, stream_radius, slot=1 - slot)
            client.set_camera(anchor)
            if not wait_for_site(client, slot, args.site_stream_timeout):
                log(f"  site {anchor.id} still loading after {args.site_stream_timeout:.0f}s; continuing")
            views = site_views(client, anchor.location, args, seed=state["route_seed"] * 100003 + index)
            t1 = time.time()
            if len(views) < args.min_views:
                log(f"  skip site {anchor.id}: {len(views)} usable cameras")
                state["sites_skipped"] += 1
                state["site_cursor"] += 1
                client.stream_site(anchor.location, 0, slot=slot)
                continue

            names = []
            shots = 0
            skipped_overlap = 0
            timing = {"respawn": 0.0, "settle": 0.0, "capture": 0.0}
            for round_index in range(args.rounds_per_site):
                if frames >= args.target or not views:
                    break
                condition = sample_condition(rng, street_yaw=anchor.rotation[1])
                names.append(condition["name"])
                apply_condition(client, condition)
                tr = time.time()
                density = respawn_traffic(client, rng, args)
                client.wait_until_settled(timeout=args.settle_timeout, min_seconds=args.respawn_wait)
                timing["respawn"] += time.time() - tr
                order = list(views)
                rng.shuffle(order)
                for view in order:
                    if frames >= args.target:
                        break
                    tv = time.time()
                    client.set_camera(view)
                    client.wait_until_settled(timeout=args.settle_timeout)
                    timing["settle"] += time.time() - tv
                    tv = time.time()
                    try:
                        shot = client.capture(pose=view, validate_pose=(round_index == 0), output_dir=str(out), **view_options)
                    except BoundlessError as error:   # failed the image-based checks (mostly sky, obstruction)
                        views.remove(view)
                        log(f"  drop {view.id}: {str(error).split(': ', 1)[-1]}")
                        continue
                    finally:
                        timing["capture"] += time.time() - tv
                    if not size_checked:
                        from PIL import Image
                        size = Image.open(shot.image_path).size
                        if size != (width, height):
                            discard_frame(out, shot.frame)
                            log(f"  image is {size[0]}x{size[1]}, not {width}x{height}; resizing the window again")
                            prepare_session(client, args)
                            continue
                        size_checked = True
                    pair = overlapping_vehicles(shot.label_path)
                    if pair:
                        # Vehicles spawned into each other are still resolving (rendered with a dissolve effect).
                        a, b = pair
                        log(f"    skip frame: {a['label']}/{a['source']}/{a['asset']} overlaps {b['label']}/{b['source']}/{b['asset']} "
                            f"at {a['distance_m']:.0f} m")
                        reject_frame(out, shot.frame, pair)
                        skipped_overlap += 1
                        continue
                    tag_label(shot, condition, density, anchor.id, round_index)
                    frames += 1
                    shots += 1
            client.stream_site(anchor.location, 0, slot=slot)
            state["sites_done"] += 1
            state["site_cursor"] += 1
            sites_since_launch += 1
            elapsed = time.time() - started
            log(f"[{frames}/{args.target}] site {anchor.id}: {len(views)} cameras x {len(names)} rounds "
                f"({'/'.join(names)}) = {shots} frames ({skipped_overlap} skipped: overlapping vehicles) | load {t1 - t0:4.1f}s, total {time.time() - t0:5.1f}s | "
                f"{(frames - frames_at_start) / max(elapsed, 1) * 3600:.0f} img/h | respawn {timing['respawn']:.0f}s, "
                f"settle {timing['settle'] / max(shots, 1):.2f}s/frame, capture {timing['capture'] / max(shots, 1):.2f}s/frame")
        except (ConnectionError, OSError, TimeoutError) as error:
            log(f"connection problem: {error!r}; restarting")
            if client is not None:
                client.close()
            client = None
            if simulator is not None:
                simulator.stop()
            time.sleep(5)
            frames = count_frames(out)
        finally:
            state_path.write_text(json.dumps(state))

    log(f"done: {frames} frames")
    if client is not None:
        client.set_weather(rain=0, snow=0, snow_cover=0, fog=0)
        if simulator is not None:
            try:
                client.quit()
            except (ConnectionError, OSError, TimeoutError, BoundlessError):
                pass
        client.close()


if __name__ == "__main__":
    main()
