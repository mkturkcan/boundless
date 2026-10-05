# Annotation format

A dataset folder contains:

| Path | Content |
|---|---|
| `rgb/NNNNNN.jpg` | The rendered image |
| `labels/NNNNNN.json` | Camera and objects of the frame (below) |
| `depth/NNNNNN.npy` | Optional depth map: float32 meters along the view axis, `inf` for sky, at half resolution by default |
| `manifest.jsonl` | One line per frame with the camera pose |

## Frame

| Field | Meaning |
|---|---|
| `frame`, `image`, `map`, `timestamp_utc`, `engine_frame` | Identification |
| `camera.location`, `camera.rotation`, `camera.quat_xyzw` | Camera pose in world coordinates (cm, degrees) |
| `camera.fov`, `camera.width`, `camera.height` | Horizontal field of view and image size |
| `camera.K` | 3x3 pinhole intrinsics (pixels) |
| `camera.world_to_camera` | 4x4 transform from world coordinates in meters to the OpenCV camera frame (x right, y down, z forward, meters) |
| `camera.pose_kind`, `camera.pose_id` | Camera type (`intersection_pole`, `midblock_pole`, `facade`, `freeway`, ...) and a stable camera id |
| `scene` | Lighting, weather and agent densities at capture time |
| `condition` | Written by `examples/collect_dataset.py`: condition name, weather, shot group and densities |

## Object

| Field | Meaning |
|---|---|
| `id` | Stable while the agent or object exists, usable for tracking across consecutive frames |
| `label` | `car`, `van`, `truck`, `bus`, `trailer`, `pedestrian`, `bicycle`, `motorcycle` |
| `parked` | `true` for parked vehicles |
| `source`, `asset` | Where the object comes from (traffic, parked, crowd or placed object) and its mesh |
| `distance_m` | Distance from the camera to the box center |
| `center`, `extent`, `rotation`, `quat_xyzw` | Oriented 3D box: center (world, cm), half size along the box's forward/right/up axes (cm), orientation |
| `corners_world` | 8 corners in world coordinates, cm |
| `corners_camera` | 8 corners in the OpenCV camera frame, meters |
| `corners_image` | 8 corners in pixels, `null` for corners behind the camera |
| `bbox_2d` | Projected 3D box, clipped to the image |
| `bbox_2d_unclipped` | Projected 3D box before clipping |
| `bbox_2d_visible` | Extent of the visible (unoccluded) part |
| `truncation` | Fraction of the projected box outside the image |
| `visible_fraction`, `visible_pixels` | Depth-tested share and estimated number of the object's pixels that are visible |
| `occlusion` | 0 visible (> 80%), 1 partly (> 40%), 2 largely occluded |

Boxes enclose the full object geometry (vehicle boxes include mirrors); pedestrian boxes have the size of a standing
person.

## Conventions

World coordinates follow Unreal Engine: centimeters, X forward, Y right, Z up (left-handed); rotations are
`[pitch, yaw, roll]` in degrees. Box corners are ordered in box-local coordinates as

```
bottom: 0 (-,-,-)  1 (+,-,-)  2 (+,+,-)  3 (-,+,-)
top:    4 (-,-,+)  5 (+,-,+)  6 (+,+,+)  7 (-,+,+)
```

with +X the object's forward direction. `boundless.labels.BOX_EDGES` lists the 12 edges.

## Helpers

```python
from boundless.labels import iter_frames, filter_objects, world_to_camera, project, BOX_EDGES

for path, frame in iter_frames("C:/data/boundless_bigcity_4k"):
    for car in filter_objects(frame, min_visible=0.3, labels=["car"]):
        center_px = project(frame, car["center"])          # (u, v) or None when behind the camera
        center_cam = world_to_camera(frame, car["center"])  # OpenCV camera frame, meters
```

`world_to_camera(frame, point_cm)` and `project(frame, point_cm)` reproduce the projection used for the labels, so
additional points (for example from your own geometry) can be mapped into the same frames.
