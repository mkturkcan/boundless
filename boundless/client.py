"""Client for the Boundless simulator API.

The game listens on TCP (default 127.0.0.1:2000). Every message is a 4-byte little-endian length
followed by UTF-8 JSON. Requests are {"id", "method", "params"}; replies are {"id", "ok", "result"}
or {"id", "ok": false, "error"}. Calls are synchronous: ``capture`` returns once the frame and its
labels are on disk.

Units follow Unreal: positions in cm (X forward, Y right, Z up, left-handed), angles in degrees,
rotations as [pitch, yaw, roll].
"""

from __future__ import annotations

import base64
import json
import socket
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence


class BoundlessError(RuntimeError):
    """The game answered with an error (bad parameters, rejected camera pose, ...)."""


@dataclass
class Pose:
    """Camera pose in Unreal world units: location in cm, rotation [pitch, yaw, roll] in degrees."""

    location: Sequence[float]
    rotation: Sequence[float] = (0.0, 0.0, 0.0)
    fov: float = 70.0
    kind: str = ""
    id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "location": [float(v) for v in self.location],
            "rotation": [float(v) for v in self.rotation],
            "fov": float(self.fov),
            "kind": self.kind,
            "id": self.id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pose":
        return cls(
            location=tuple(data["location"]),
            rotation=tuple(data.get("rotation", (0.0, 0.0, 0.0))),
            fov=float(data.get("fov", 70.0)),
            kind=data.get("kind", ""),
            id=data.get("id", ""),
        )


@dataclass
class CaptureResult:
    """What ``Client.capture`` returns. ``annotations`` is the same JSON as labels/<frame>.json."""

    raw: Dict[str, Any]
    frame: str = ""
    output_dir: str = ""
    image_path: Optional[str] = None
    label_path: Optional[str] = None
    depth_path: Optional[str] = None
    pose: Optional[Pose] = None
    annotations: Optional[Dict[str, Any]] = None
    image_bytes: Optional[bytes] = field(default=None, repr=False)

    @property
    def objects(self) -> List[Dict[str, Any]]:
        return (self.annotations or {}).get("objects", [])

    @property
    def camera(self) -> Dict[str, Any]:
        return (self.annotations or {}).get("camera", {})

    def image(self):
        """Decoded image as an HxWx3 uint8 numpy array (needs numpy and Pillow)."""
        import io

        import numpy as np
        from PIL import Image

        if self.image_bytes is not None:
            return np.asarray(Image.open(io.BytesIO(self.image_bytes)).convert("RGB"))
        if self.image_path:
            return np.asarray(Image.open(self.image_path).convert("RGB"))
        raise ValueError("capture had no image (use return_image=True or save_rgb)")

    def depth(self):
        """Depth in meters (float32, inf = sky), if the capture saved it."""
        import numpy as np

        if not self.depth_path:
            raise ValueError("capture had no depth (pass save_depth=True)")
        return np.load(self.depth_path)


def _pose_param(pose: Any) -> Dict[str, Any]:
    if isinstance(pose, Pose):
        return pose.to_dict()
    if isinstance(pose, dict):
        return pose
    raise TypeError("pose must be a Pose or a dict with location/rotation/fov")


def _drop_none(values: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in values.items() if v is not None}


class Client:
    """Connection to a running game. Use as a context manager or call ``close()``.

    >>> with Client() as client:
    ...     client.load_map("Small City")
    ...     pose = client.sample_camera_poses(n=1, seed=3)[0]
    ...     shot = client.capture(pose=pose, validate_pose=True)
    ...     print(shot.image_path, len(shot.objects))
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 2000, timeout: float = 300.0, connect_timeout: float = 120.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._next_id = 1
        self._sock: Optional[socket.socket] = None
        self._connect(connect_timeout)

    # -- connection -------------------------------------------------------------------------

    def _connect(self, connect_timeout: float) -> None:
        deadline = time.monotonic() + connect_timeout
        last_error: Optional[Exception] = None
        while time.monotonic() < deadline:
            try:
                sock = socket.create_connection((self.host, self.port), timeout=5.0)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._sock = sock
                return
            except OSError as exc:  # the game may still be starting
                last_error = exc
                time.sleep(1.0)
        raise ConnectionError(f"could not reach the game at {self.host}:{self.port}: {last_error}")

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _recv_exact(self, size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining > 0:
            chunk = self._sock.recv(min(remaining, 1 << 20))
            if not chunk:
                raise ConnectionError("the game closed the connection")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def call(self, method: str, timeout: Optional[float] = None, **params: Any) -> Dict[str, Any]:
        """Low-level call: sends one request and waits for its reply."""
        if self._sock is None:
            raise ConnectionError("client is closed")
        request_id = self._next_id
        self._next_id += 1
        payload = json.dumps({"id": request_id, "method": method, "params": params}).encode("utf-8")
        self._sock.settimeout(timeout if timeout is not None else self.timeout)
        self._sock.sendall(struct.pack("<I", len(payload)) + payload)
        while True:
            (length,) = struct.unpack("<I", self._recv_exact(4))
            reply = json.loads(self._recv_exact(length).decode("utf-8"))
            if reply.get("id") != request_id:
                continue  # a late reply to an earlier call that timed out on our side
            if not reply.get("ok", False):
                raise BoundlessError(f"{method}: {reply.get('error', 'unknown error')}")
            return reply.get("result", {})

    # -- session and maps -------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return self.call("get_status")

    def ping(self) -> Dict[str, Any]:
        return self.call("ping")

    def list_maps(self) -> List[Dict[str, str]]:
        return self.call("list_maps")["maps"]

    def load_map(self, map: str, wait: bool = True, timeout: float = 1800.0) -> Dict[str, Any]:
        """Open a map by display name ("Small City"), short name or /Game path. Waits until it is ready."""
        return self.call("load_map", timeout=timeout, map=map, wait=wait)

    def wait_until_ready(self, timeout: float = 1800.0, poll: float = 2.0) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.status()
            if status.get("world_ready"):
                return status
            time.sleep(poll)
        raise TimeoutError("level did not become ready")

    def get_session(self) -> Dict[str, Any]:
        return self.call("get_session")

    def set_session(self, apply: bool = True, respawn: bool = False, save: bool = False, **settings: Any) -> Dict[str, Any]:
        """Partially update the session settings, e.g. set_session(capture={"save_depth": True})."""
        return self.call("set_session", settings=settings, apply=apply, respawn=respawn, save=save)

    def quit(self) -> None:
        try:
            self.call("quit", timeout=10.0)
        finally:
            self.close()

    # -- scene ------------------------------------------------------------------------------

    def set_weather(self, rain: float = None, snow: float = None, snow_cover: float = None, fog: float = None,
                    wetness: float = None) -> Dict[str, Any]:
        """Weather around the camera, each value 0..1; omitted values keep their current setting.

        rain: falling rain (0.1 = drizzle, 1 = downpour). snow: falling snow. snow_cover: snow lying on the ground.
        fog: height-fog density added to the map's own.
        """
        return self.call("set_weather", **_drop_none(dict(rain=rain, snow=snow, snow_cover=snow_cover, fog=fog, wetness=wetness)))

    def set_lighting(self, sun_elevation: float = None, sun_azimuth: float = None, sun_intensity_scale: float = None,
                     temperature: float = None, sky_intensity_scale: float = None, **more: Any) -> Dict[str, Any]:
        """Sun, sky and camera exposure. Omitted values keep their current setting.

        Scales are relative to the map's authored values for the current mode. ``more``:
        night (bool; lit windows and street lights, the directional light becomes the moon), sky_dome_intensity,
        sky_dome_tint [r, g, b], sky_sun_intensity, exposure_bias (EV), auto_exposure (bool),
        auto_exposure_range [min, max] (EV100), white_balance (K, 0 = default), saturation, contrast.
        """
        return self.call("set_lighting", **_drop_none(dict(
            sun_elevation=sun_elevation, sun_azimuth=sun_azimuth, sun_intensity_scale=sun_intensity_scale,
            temperature=temperature, sky_intensity_scale=sky_intensity_scale, **more)))

    def reset_lighting(self) -> Dict[str, Any]:
        return self.call("reset_lighting")

    def set_density(self, traffic: float = None, parked: float = None, crowd: float = None, respawn: bool = True) -> Dict[str, Any]:
        """Multipliers on the map's authored Mass spawn counts (1.0 = as authored)."""
        return self.call("set_density", respawn=respawn, **_drop_none(dict(traffic=traffic, parked=parked, crowd=crowd)))

    def randomize_scene(self, lighting: bool = None, weather: bool = None, traffic: bool = None, seed: int = None,
                        ranges: Dict[str, Any] = None) -> Dict[str, Any]:
        """Draw a new scene. ``ranges`` overrides the randomization ranges (see README)."""
        return self.call("randomize_scene", **_drop_none(dict(lighting=lighting, weather=weather, traffic=traffic, seed=seed, ranges=ranges)))

    def get_scene(self) -> Dict[str, Any]:
        return self.call("get_scene")

    # -- cameras ----------------------------------------------------------------------------

    def sample_camera_poses(self, n: int = 10, seed: int = None, kinds: Iterable[str] = None,
                            near: Sequence[float] = None, radius: float = None, rig: Dict[str, Any] = None) -> List[Pose]:
        """Generate infrastructure-style camera poses from the road network.

        kinds: any of "intersection", "midblock", "facade", "freeway". near/radius (cm) restrict them to a region.
        "facade" = building windows on floors min_floor..max_floor, mounted on the wall at capture time.
        rig overrides the camera rig (height_range, fov_range, look_distance_range, weights, snap_to_poles...).
        Poses are validated (and possibly moved onto a pole) when captured with validate_pose=True.
        """
        result = self.call("sample_camera_poses", **_drop_none(dict(
            n=n, seed=seed, kinds=list(kinds) if kinds else None, near=list(near) if near else None, radius=radius, rig=rig)))
        return [Pose.from_dict(p) for p in result["poses"]]

    def set_camera(self, pose: Any = None, location: Sequence[float] = None, rotation: Sequence[float] = None, fov: float = None) -> Pose:
        """View through the infrastructure camera at a pose (the drone is hidden until release_camera)."""
        if pose is None:
            pose = Pose(location=location, rotation=rotation or (0.0, 0.0, 0.0), fov=fov or 70.0)
        return Pose.from_dict(self.call("set_camera", pose=_pose_param(pose)))

    def release_camera(self) -> None:
        self.call("release_camera")

    def get_camera(self) -> Dict[str, Any]:
        return self.call("get_camera")

    def validate_pose(self, pose: Any) -> Dict[str, Any]:
        """Geometry checks for a pose whose surroundings are already loaded. Returns valid/reason/pose."""
        return self.call("validate_pose", pose=_pose_param(pose))

    def save_pose(self, pose: Any = None) -> Dict[str, Any]:
        """Append a pose (default: the current view) to Saved/Boundless/poses_<map>.json."""
        return self.call("save_pose", **({"pose": _pose_param(pose)} if pose is not None else {}))

    def list_saved_poses(self) -> List[Pose]:
        return [Pose.from_dict(p) for p in self.call("list_saved_poses")["poses"]]

    # -- capture ----------------------------------------------------------------------------

    def capture(self, pose: Any = None, validate_pose: bool = False, output_dir: str = None, name: str = None,
                return_image: bool = False, return_annotations: bool = True, write_files: bool = True,
                randomize: Dict[str, bool] = None, timeout: float = None, **options: Any) -> CaptureResult:
        """Render one frame and label it.

        pose: move the infrastructure camera there first (None = capture the current view).
        validate_pose: snap to a pole / reject bad poses once the area is streamed in (raises BoundlessError).
        randomize: e.g. {"lighting": True, "weather": True, "traffic": True}, applied before the shot.
        options: capture options, e.g. save_depth=True, settle_seconds=3, image_format="jpg",
                 max_distance=20000 (cm), min_visible_fraction=0.1, freeze_mode="Pause".
        """
        params: Dict[str, Any] = dict(
            validate_pose=validate_pose, return_image=return_image, return_annotations=return_annotations,
            write_files=write_files, options=options)
        if pose is not None:
            params["pose"] = _pose_param(pose)
        params.update(_drop_none(dict(output_dir=output_dir, name=name, randomize=randomize)))
        raw = self.call("capture", timeout=timeout, **params)
        image_bytes = base64.b64decode(raw["image_base64"]) if "image_base64" in raw else None
        return CaptureResult(
            raw=raw,
            frame=raw.get("frame", ""),
            output_dir=raw.get("output_dir", ""),
            image_path=raw.get("image_path"),
            label_path=raw.get("label_path"),
            depth_path=raw.get("depth_path"),
            pose=Pose.from_dict(raw["pose"]) if "pose" in raw else None,
            annotations=raw.get("annotations"),
            image_bytes=image_bytes,
        )

    def get_annotations(self, **options: Any) -> Dict[str, Any]:
        """Labels for the current view without rendering an image."""
        return self.call("get_annotations", options=options)

    def start_auto(self, num_poses: int = None, shots_per_pose: int = None) -> Dict[str, Any]:
        """Start the in-game auto-capture loop (same as F7)."""
        return self.call("start_auto", **_drop_none(dict(num_poses=num_poses, shots_per_pose=shots_per_pose)))

    def stop_auto(self) -> Dict[str, Any]:
        return self.call("stop_auto")

    # -- engine access ----------------------------------------------------------------------

    def console(self, command: str) -> str:
        """Run a console command, e.g. "r.ScreenPercentage 100" or "sg.ShadowQuality 3"."""
        return self.call("console", command=command).get("output", "")

    def set_time_dilation(self, value: float) -> None:
        self.call("set_time_dilation", value=value)

    def pause(self, paused: bool = True) -> None:
        """Pause the whole game (the camera cannot move while paused)."""
        self.call("pause", paused=paused)

    def stream_site(self, location: Sequence[float], radius: float, slot: int = 0) -> bool:
        """Keep everything within ``radius`` cm of ``location`` loaded (``radius=0`` releases it). Slot 0 is the current
        site, slot 1 can preload the next one. Returns whether the area is fully loaded."""
        return self.call("stream_site", location=list(location), radius=radius, slot=slot)["ready"]

    def readiness(self, radius: float = 25000.0, near_radius: float = 8000.0) -> Dict[str, Any]:
        """Whether the current view has finished loading. ``missing_total`` counts agents within ``radius`` cm of the
        camera that should be on screen but are not spawned yet (``missing`` by kind); ``near_actors_pending`` counts
        agents within ``near_radius`` still waiting for their full actor. Also ``streaming_complete`` and
        ``shaders_remaining``."""
        return self.call("readiness", radius=radius, near_radius=near_radius)

    def wait_until_settled(self, timeout: float = 30.0, min_seconds: float = 0.0, plateau_polls: int = 4,
                           poll: float = 0.2) -> float:
        """Block until the view has finished loading and at least ``min_seconds`` have passed: no nearby agent waiting
        for its actor, streaming complete, and the number of agents still missing in view no longer decreasing over
        ``plateau_polls`` consecutive polls (a few agents can stay missing for a while when their spawn is retried).
        Returns the seconds waited."""
        start = time.time()
        history: List[int] = []
        while time.time() - start < timeout:
            state = self.readiness()
            missing = int(state.get("missing_total", 0))
            history.append(missing)
            loaded = state.get("near_actors_pending", 0) == 0 and state.get("streaming_complete", True)
            recent = history[-plateau_polls:]
            plateau = len(recent) == plateau_polls and (missing == 0 or min(recent) >= recent[0])
            if loaded and plateau and time.time() - start >= min_seconds:
                break
            time.sleep(poll)
        return time.time() - start

    def set_actor_spawn_budget(self, seconds: float) -> float:
        """Time per frame the simulator may spend spawning agent actors (vehicles, pedestrians) near the camera.
        Raising it from the gameplay default makes agents appear within a few frames after a camera move or re-spawn."""
        return self.call("set_actor_spawn_budget", seconds=seconds)["seconds"]

    def site_ready(self, slot: int = 0) -> bool:
        """Whether the area requested with ``stream_site`` is fully loaded."""
        return self.call("stream_site", slot=slot)["ready"]
