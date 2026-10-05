# Python API reference

```python
from boundless import Client, Pose, CaptureResult, BoundlessError
```

The simulator listens on TCP (default `127.0.0.1:2000`; change it with `-BoundlessPort=` and allow other machines with
`-BoundlessHost=0.0.0.0`). Calls are synchronous: `capture` returns once the image and its labels are written.

**Units.** Unreal Engine conventions throughout: positions in centimeters, angles in degrees, rotations as
`[pitch, yaw, roll]`, world axes X forward, Y right, Z up (left-handed). Label files additionally provide every box in
the OpenCV camera frame (meters) and in pixels.

## Launching the simulator

```bat
Boundless.exe -BoundlessMode=api -BoundlessMap=/Game/Map/Big_City_LVL -windowed -ResX=3840 -ResY=2160 -ForceRes
```

| Flag | Meaning |
|---|---|
| `-BoundlessMode=api` | Start directly in API mode (skips the main menu). `auto` starts the built-in capture loop. |
| `-BoundlessMap=<path>` | Map to load: `/Game/Map/Small_City_LVL` or `/Game/Map/Big_City_LVL` |
| `-BoundlessPort=<n>` | API port (default 2000) |
| `-BoundlessHost=<address>` | Bind address (default `127.0.0.1`) |
| `-BoundlessSeed=<n>` | Random seed for the session |
| `-BoundlessOut=<folder>` | Default output folder |
| `-ResX`, `-ResY`, `-windowed`, `-ForceRes` | Image size. Images are captured at the window size; `-ForceRes` allows windows larger than the desktop (e.g. 4K on a 1080p monitor). |

Without `-BoundlessMode` the simulator opens a main menu for choosing the map, mode, resolution and scene settings.

## Client

```python
Client(host="127.0.0.1", port=2000, timeout=300.0, connect_timeout=120.0)
```

Usable as a context manager. `call(method, **params)` sends any request directly.

### Session and maps

| Method | Returns |
|---|---|
| `status()` | `map`, `map_path`, `world_ready`, `mode`, `seed`, `output_dir`, `shaders_remaining`, current scene |
| `wait_until_ready(timeout=1800)` | Blocks until a map is loaded and ready |
| `list_maps()` | `[{"name": "Small City", "path": ...}, {"name": "Big City", ...}]` |
| `load_map(map, wait=True)` | Loads by display name, short name or path |
| `get_session()`, `set_session(**settings)` | The full session (capture options, camera rig, randomization ranges); partial updates are allowed, e.g. `set_session(capture={"max_distance": 50000})` |
| `quit()` | Closes the simulator |

### Scene

**`set_lighting(**fields)`.** Fields not passed keep their current value. Intensity scales are relative to the
map's authored values for the current mode (day or night).

| Field | Meaning |
|---|---|
| `sun_elevation`, `sun_azimuth` | Sun (or moon at night) direction, degrees. The sky follows the sun. |
| `sun_intensity_scale` | Sun intensity multiplier (about 0.01-0.2 overcast, 2-8 sunny) |
| `temperature` | Sun color temperature, Kelvin |
| `sky_intensity_scale` | Sky light multiplier |
| `night` | `True` switches to the night setup: lit windows and street lights, dark sky; the directional light becomes the moon |
| `sky_dome_intensity`, `sky_dome_tint` | Sky brightness and `[r, g, b]` tint multipliers (grey-blue tints give overcast skies) |
| `sky_sun_intensity` | Brightness of the sun disk in the sky (0 for overcast) |
| `auto_exposure`, `auto_exposure_range` | Histogram auto exposure within `[min, max]` EV100, like a real camera |
| `exposure_bias` | Exposure compensation in EV (about -2.5 to -4 for night) |
| `white_balance` | Camera white balance in Kelvin (0 = default; 3000-4800 neutralizes sodium street lights at night) |
| `saturation`, `contrast` | Color grading multipliers |

`reset_lighting()` restores the map's authored lighting.

**`set_weather(rain=None, snow=None, snow_cover=None, fog=None)`.** Values from 0 to 1, applied around the camera and
following it. `rain` is falling rain (0.05 drizzle to 1 downpour), `snow` is falling snow, `snow_cover` is snow lying on
the ground, and `fog` adds height-fog density.

**`set_density(traffic=None, parked=None, crowd=None, respawn=True)`.** Multipliers on the map's default counts of
moving vehicles, parked vehicles and pedestrians. With `respawn=True` all agents are removed and spawned again at new
places; allow two or three seconds for them to initialize before capturing.

**`randomize_scene(lighting=None, weather=None, traffic=None, seed=None, ranges=None)`.** Draws a new scene from the
session's randomization ranges (`ranges` overrides them, e.g. `{"sun_elevation_range": [5, 40]}`).

`get_scene()` returns the current lighting, weather and densities.

### Cameras

**`sample_camera_poses(n=10, seed=None, kinds=None, near=None, radius=None, rig=None) -> list[Pose]`.** Generates
infrastructure camera poses from the road network. `kinds` is any subset of `intersection`, `midblock`, `facade` and
`freeway`; `near`/`radius` (cm) restrict them to a region. Poses are provisional until captured with
`validate_pose=True`, which mounts them (pole snapping for pole cameras, the wall for facade cameras) and rejects poses
that fail the geometry checks (inside geometry, view blocked, mostly sky, obstruction in front of the lens).

`rig` overrides the camera rig:

| Field | Default | Meaning |
|---|---|---|
| `height_range` | `[450, 1100]` | Mounting height above the road, cm |
| `fov_range` | `[60, 90]` | Horizontal field of view, degrees |
| `look_distance_range` | `[1500, 5000]` | Distance along the road to the aim point, cm |
| `intersection_weight`, `mid_block_weight`, `freeway_weight`, `facade_weight` | `0.55`, `0.35`, `0.1`, `0.3` | Relative frequency of each kind |
| `snap_to_poles`, `pole_search_radius` | `true`, `1200` | Mount pole cameras on a nearby traffic-light or street-light pole |
| `min_floor`, `max_floor`, `floor_height` | `2`, `3`, `350` | Facade camera floors (1 = ground floor) |
| `window_height_range` | `[100, 170]` | Facade camera height above its floor, cm |
| `facade_pitch_range` | `[-30, -5]` | Facade camera pitch, degrees |
| `facade_yaw_jitter`, `facade_roll_jitter` | `35`, `2` | Maximum yaw away from the wall normal and roll, degrees |
| `facade_offset_range` | `[40, 90]` | Distance of the lens from the wall, cm |
| `prefer_windows` | `true` | Prefer glazed parts of the facade |

| Method | |
|---|---|
| `set_camera(pose=None, location=None, rotation=None, fov=None) -> Pose` | Views through an infrastructure camera |
| `release_camera()` | Returns to the free camera |
| `get_camera()` | Current pose plus `view` (K, extrinsics, image size) |
| `validate_pose(pose)` | Geometry checks for a pose whose surroundings are loaded; returns `valid`, `reason`, refined `pose` |
| `save_pose(pose=None)`, `list_saved_poses()` | Per-map list of saved poses |

`Pose(location, rotation, fov, kind, id)` has `to_dict()` and `Pose.from_dict()`.

### Capture

**`capture(pose=None, validate_pose=False, output_dir=None, name=None, return_image=False, return_annotations=True,
write_files=True, randomize=None, timeout=None, **options) -> CaptureResult`**

Moves the camera (when `pose` is given), waits for the scene to settle, renders one frame and labels it. `randomize`
(e.g. `{"lighting": True, "weather": True, "traffic": True}`) draws a new scene first. Raises `BoundlessError` when a
pose is rejected.

| Option | Default | Meaning |
|---|---|---|
| `settle_frames`, `settle_seconds` | `45`, `2.0` | Rendered frames and seconds to wait after moving the camera |
| `streaming_timeout_seconds` | `30` | Maximum wait for the city around the camera to load |
| `image_format`, `jpeg_quality` | `"png"`, `95` | `"png"` or `"jpg"` |
| `max_distance` | `15000` | Objects farther than this are not labeled, cm (the dataset generator uses 200000) |
| `min_visible_fraction`, `min_visible_pixels` | `0.05`, `64` | Visibility thresholds; pixels are counted at depth-map resolution |
| `depth_downscale` | `2` | Depth map resolution is 1/N of the image |
| `save_rgb`, `save_annotations`, `save_depth` | `true`, `true`, `false` | Files to write; depth is `depth/<frame>.npy`, float32 meters |
| `include_mass`, `include_actors`, `include_static_meshes` | `true` | Object sources to label (agents, actors, placed vehicles) |
| `freeze_mode`, `freeze_frames` | `"None"`, `12` | `"Pause"` pauses the world for `freeze_frames` before the shot |

`CaptureResult` provides `frame`, `image_path`, `label_path`, `depth_path`, `pose` (the final, mounted pose),
`annotations`, `objects`, `camera`, `image()` (numpy array) and `depth()`.

`get_annotations(**options)` returns the labels of the current view without rendering an image.

### Engine

| Method | |
|---|---|
| `console(command)` | Runs a console command, e.g. `console("r.ScreenPercentage 100")` |
| `pause(paused=True)`, `set_time_dilation(value)` | Simulation control |
| `start_auto(num_poses, shots_per_pose)`, `stop_auto()` | The built-in capture loop |

## Wire protocol

Every message is a 4-byte little-endian length followed by UTF-8 JSON. Requests are
`{"id": 1, "method": "capture", "params": {...}}`; replies are `{"id": 1, "ok": true, "result": {...}}` or
`{"id": 1, "ok": false, "error": "..."}`. Parameter names are the snake_case names used above.
