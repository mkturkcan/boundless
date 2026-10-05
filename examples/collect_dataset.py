"""Unattended, resumable dataset generation with varied time of day, weather and traffic.

For every infrastructure camera (an intersection-heavy mix of pole, mid-block, facade and freeway cameras):

  * ``--segments-per-pose`` shot groups. Each group draws a new condition (sunny, low sun, overcast, rain, snow,
    dusk, night, night rain; see ``CONDITIONS``), respawns moving traffic, parked vehicles and pedestrians at new
    densities, and waits for them to initialize;
  * ``--shots-per-segment`` shots per group while the traffic moves on.

The simulator is launched (and relaunched after a crash) when ``--launch`` is given, and a run can be interrupted and
resumed at any time. The drawn condition is stored in every label file under ``"condition"``.

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


def day_lighting(rng: random.Random, **override) -> dict:
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


def sample_condition(rng: random.Random, name: str | None = None) -> dict:
    if name is None:
        pick = rng.random()
        for name, probability in CONDITIONS:
            pick -= probability
            if pick <= 0:
                break
    weather = dict(rain=0.0, snow=0.0, snow_cover=0.0, fog=0.0)
    if name == "sunny":
        lighting = day_lighting(rng)
        weather["fog"] = U(rng, 0.0, 0.15)
    elif name == "low_sun":
        lighting = day_lighting(rng, sun_elevation=U(rng, 2, 12), temperature=U(rng, 2800, 4500),
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
        weather.update(rain=log_uniform(rng, 0.08, 1.0), fog=U(rng, 0.15, 0.7))
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


def tag_label(shot, condition: dict, density: dict, segment: int) -> None:
    """Store the drawn condition in the frame's label file."""
    if not shot.label_path:
        return
    path = Path(shot.label_path)
    label = json.loads(path.read_text(encoding="utf-8"))
    label["condition"] = dict(name=condition["name"], weather=condition["weather"], segment=segment, density=density)
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
            tag_label(shot, condition, density, 0)
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

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target", type=int, default=10000)
    parser.add_argument("--map", default="Big City", help='"Big City", "Small City" or a map path')
    parser.add_argument("--resolution", default="3840x2160")
    parser.add_argument("--kinds", default="intersection,midblock,facade,freeway")
    parser.add_argument("--candidates", type=int, default=6000, help="camera poses drawn from the road network per route")
    parser.add_argument("--route-poses", type=int, default=1500, help="city-wide subset visited along a short tour")
    parser.add_argument("--segments-per-pose", type=int, default=3, help="condition + traffic respawn groups per camera")
    parser.add_argument("--shots-per-segment", type=int, default=4)
    parser.add_argument("--shot-interval", type=float, default=0.6, help="seconds of traffic motion between shots of a group")
    parser.add_argument("--respawn-wait", type=float, default=2.5, help="seconds for respawned agents to initialize")
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
    parser.add_argument("--project", default=None, help="only for editor builds: the .uproject passed to the editor")
    args = parser.parse_args()

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
    state = json.loads(state_path.read_text()) if state_path.exists() else {"pose_cursor": 0, "accepted": 0, "rejected": 0, "route_seed": args.seed}
    rng = random.Random(args.seed + state["accepted"] * 7919)
    started = time.time()
    client = None
    route: list[Pose] = []
    width, height = (int(v) for v in args.resolution.lower().split("x"))
    frames = count_frames(out)
    frames_at_start = frames
    options_first = capture_options(args, settle_seconds=args.respawn_wait + 1.0, settle_frames=40)
    options_group = capture_options(args, settle_seconds=args.respawn_wait + 0.8, settle_frames=30, streaming_timeout_seconds=10.0)
    options_next = capture_options(args, settle_seconds=args.shot_interval, settle_frames=4, freeze_frames=1, streaming_timeout_seconds=5.0)
    size_checked = False

    while frames < args.target:
        if args.max_hours and time.time() - started > args.max_hours * 3600:
            log(f"time limit reached: {frames} frames")
            break
        try:
            if client is None:
                frames = compact_dataset(out)
                client = connect(args, simulator)
                prepare_session(client, args)
                size_checked = False
                route = build_route(client, args, state["route_seed"])
                log(f"{len(route)} camera poses on the route, resuming at {state['pose_cursor']}")
            if state["pose_cursor"] >= len(route):
                state["pose_cursor"] = 0
                state["route_seed"] += 1
                route = build_route(client, args, state["route_seed"])
            pose = route[state["pose_cursor"]]
            state["pose_cursor"] += 1

            t0 = time.time()
            condition = sample_condition(rng)
            apply_condition(client, condition)
            density = respawn_traffic(client, rng, args)
            try:
                first = client.capture(pose=pose, validate_pose=True, output_dir=str(out), **options_first)
            except BoundlessError as error:
                state["rejected"] += 1
                log(f"  skip {pose.id}: {str(error).split(': ', 1)[-1]}")
                continue
            refined = first.pose or pose
            if not size_checked:
                from PIL import Image
                size = Image.open(first.image_path).size
                if size != (width, height):
                    discard_frame(out, first.frame)
                    log(f"  image is {size[0]}x{size[1]}, not {width}x{height}; resizing the window again")
                    prepare_session(client, args)
                    continue
                size_checked = True
            if not -45.0 <= refined.rotation[0] <= 5.0:
                discard_frame(out, first.frame)
                state["rejected"] += 1
                log(f"  skip {refined.id}: pitch {refined.rotation[0]:.0f} deg after mounting")
                continue
            state["accepted"] += 1
            frames += 1
            tag_label(first, condition, density, 0)
            names = [condition["name"]]
            shots = 1
            t1 = time.time()
            for segment in range(args.segments_per_pose):
                if segment > 0:
                    if frames >= args.target:
                        break
                    condition = sample_condition(rng)
                    names.append(condition["name"])
                    apply_condition(client, condition)
                    density = respawn_traffic(client, rng, args)
                    shot = client.capture(pose=refined, output_dir=str(out), **options_group)
                    tag_label(shot, condition, density, segment)
                    frames += 1
                    shots += 1
                for _ in range(args.shots_per_segment - 1):
                    if frames >= args.target:
                        break
                    shot = client.capture(pose=refined, output_dir=str(out), **options_next)
                    tag_label(shot, condition, density, segment)
                    frames += 1
                    shots += 1
            t2 = time.time()
            elapsed = time.time() - started
            log(f"[{frames}/{args.target}] {refined.kind:>17} {first.raw.get('num_objects', 0):3d} objects "
                f"{'/'.join(names):>28} | first {t1 - t0:4.1f}s, then {(t2 - t1) / max(shots - 1, 1):4.2f}s/shot | "
                f"{(frames - frames_at_start) / max(elapsed, 1) * 3600:.0f} img/h")
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
