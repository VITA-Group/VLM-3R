"""Recompute VSTI-Bench ground-truth labels affected by the camera-position bug.

The original generators extracted the camera center from a camera-to-world pose
with ``-R.T @ t`` (the w2c translation) instead of ``pose[:3, 3]``. This script
recomputes, in place, the labels of the three affected question families
(camera_displacement, camera_obj_abs_dist, camera_obj_rel_dist_v1/v2/v3) for an
already-released test.json, keeping the question set unchanged so results stay
comparable. Items whose corrected geometry violates the generators' original
validity filters are flagged for removal rather than silently dropped.

Inputs are the released test.json plus the exact metadata used at generation
time (scannet_frame_metadata_val.json / scannet_metadata_val.json from
Journey9ni/scannet-metadata).

Usage:
    python fix_vstibench_gt_labels.py \
        --test_json test.json \
        --frame_meta scannet_frame_metadata_val.json \
        --scene_meta scannet_metadata_val.json \
        --output_dir corrected/
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False

# Validity filters used by the original generators.
DISPLACEMENT_RANGE = (0.2, 10.0)   # CameraDisplacementQAGenerator
ABS_DIST_RANGE = (0.2, 10.0)       # AbsoluteDistanceQAGenerator
REL_DIST_MIN = 0.5                 # RelativeDistanceQAGenerator min_dist_threshold
REL_DIST_AMBIGUITY = 0.15          # RelativeDistanceQAGenerator ambiguity_threshold

DISPLACEMENT_PAT = re.compile(r"frame (\d+) and frame (\d+) of (\d+)")
ABS_DIST_PAT = re.compile(r"nearest point of the (.+?) in frame (\d+) of (\d+)")
REL_DIST_PAT = re.compile(r"frame (\d+) of (\d+)")

OPTION_LETTER_PAT = re.compile(r"^([A-D])\.\s*(.+)$")


def sample_points_in_bbox(centroid, normalized_axes, axes_lengths, distance=0.05):
    """Mirror of common_utils.sample_points_in_oriented_bbox_uniform.

    Deterministic 5 cm grid inside the OBB, hollowed center, cropped to the
    box. Uses open3d when available (bit-identical to generation); otherwise
    emulates the crop by re-projecting world points into the box frame.
    """
    center = np.asarray(centroid, dtype=np.float64)
    R = np.asarray(normalized_axes, dtype=np.float64).reshape(3, 3).T
    extent = np.asarray(axes_lengths, dtype=np.float64)

    nx, ny, nz = (int(np.ceil(e / distance)) for e in extent)
    x = np.linspace(-extent[0] / 2, extent[0] / 2, nx)
    y = np.linspace(-extent[1] / 2, extent[1] / 2, ny)
    z = np.linspace(-extent[2] / 2, extent[2] / 2, nz)
    xx, yy, zz = np.meshgrid(x, y, z)
    pts = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    pts = pts[np.any(np.abs(pts) > extent / 4, axis=1)]
    if pts.shape[0] == 0:
        return sample_points_in_bbox(centroid, normalized_axes, axes_lengths,
                                     distance * 0.5)
    world = pts @ R.T + center
    if HAS_O3D:
        # Same crop the generators applied via
        # common_utils.sample_points_in_oriented_bbox_uniform.
        bbox = o3d.geometry.OrientedBoundingBox(
            center.reshape(3, 1), R, extent.reshape(3, 1))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(world)
        pcd = pcd.crop(bbox)
        if len(pcd.points) == 0:
            return sample_points_in_bbox(centroid, normalized_axes,
                                         axes_lengths, distance * 0.5)
        return np.asarray(pcd.points)
    # Emulate o3d's crop: points landing epsilon outside after the world
    # round-trip are dropped, matching generation-time behavior closely.
    local = (world - center) @ R
    world = world[np.all(np.abs(local) <= extent / 2, axis=1)]
    if world.shape[0] == 0:
        return sample_points_in_bbox(centroid, normalized_axes, axes_lengths,
                                     distance * 0.5)
    return world


class SceneCache:
    def __init__(self, frame_meta, scene_meta):
        self.frame_meta = frame_meta
        self.scene_meta = scene_meta
        self._valid_frames = {}
        self._inst_map = {}
        self._points = {}

    def valid_frames(self, scene):
        if scene not in self._valid_frames:
            frames = self.frame_meta[scene]["frames"]
            self._valid_frames[scene] = sorted(
                [f for f in frames if f.get("camera_pose_camera_to_world")],
                key=lambda x: x["frame_id"])
        return self._valid_frames[scene]

    def instance_map(self, scene):
        if scene not in self._inst_map:
            m = {}
            for cat, lst in self.scene_meta[scene].get("object_bboxes", {}).items():
                for o in lst:
                    iid = o.get("instance_id")
                    if iid is not None and all(
                            k in o for k in ("centroid", "normalizedAxes", "axesLengths")):
                        m[iid] = (cat, o)
            self._inst_map[scene] = m
        return self._inst_map[scene]

    def instance_points(self, scene, iid):
        key = (scene, iid)
        if key not in self._points:
            _, meta = self.instance_map(scene)[iid]
            self._points[key] = sample_points_in_bbox(
                meta["centroid"], meta["normalizedAxes"], meta["axesLengths"])
        return self._points[key]

    def frame_unique_instance(self, scene, frame, category):
        """instance_id if `category` has exactly one visible instance in frame."""
        inst_map = self.instance_map(scene)
        ids = [b["instance_id"] for b in frame.get("bboxes_2d", [])
               if b.get("instance_id") in inst_map
               and inst_map[b["instance_id"]][0] == category]
        return ids[0] if len(ids) == 1 else None


def camera_center(pose_list):
    P = np.asarray(pose_list, dtype=np.float64)
    return P[:3, 3]


def scene_of(item):
    return item["video_path"].split("/")[-1].replace(".mp4", "")


def fix_displacement(item, cache):
    m = DISPLACEMENT_PAT.search(item["question"])
    i, j, _ = map(int, m.groups())
    valid = cache.valid_frames(scene_of(item))
    c0 = camera_center(valid[i - 1]["camera_pose_camera_to_world"])
    c1 = camera_center(valid[j - 1]["camera_pose_camera_to_world"])
    new_gt = round(float(np.linalg.norm(c1 - c0)), 1)
    drop = not (DISPLACEMENT_RANGE[0] <= new_gt <= DISPLACEMENT_RANGE[1])
    return new_gt, None, ("out_of_range" if drop else None)


def fix_abs_dist(item, cache):
    m = ABS_DIST_PAT.search(item["question"])
    category, k, _ = m.group(1), int(m.group(2)), int(m.group(3))
    scene = scene_of(item)
    frame = cache.valid_frames(scene)[k - 1]
    iid = cache.frame_unique_instance(scene, frame, category)
    if iid is None:
        return None, None, "object_not_unique_in_frame"
    cam = camera_center(frame["camera_pose_camera_to_world"])
    pts = cache.instance_points(scene, iid)
    new_gt = round(float(np.linalg.norm(pts - cam, axis=1).min()), 1)
    drop = not (ABS_DIST_RANGE[0] <= new_gt <= ABS_DIST_RANGE[1])
    return new_gt, None, ("out_of_range" if drop else None)


def fix_rel_dist(item, cache):
    m = REL_DIST_PAT.search(item["question"])
    k = int(m.group(1))
    scene = scene_of(item)
    frame = cache.valid_frames(scene)[k - 1]
    cam = camera_center(frame["camera_pose_camera_to_world"])

    letters, categories = [], []
    for opt in item["options"]:
        lm = OPTION_LETTER_PAT.match(opt.strip())
        letters.append(lm.group(1))
        categories.append(lm.group(2).strip())

    dists = []
    for cat in categories:
        iid = cache.frame_unique_instance(scene, frame, cat)
        if iid is None:
            return None, None, "option_not_unique_in_frame"
        pts = cache.instance_points(scene, iid)
        # The generators compared 0.1-rounded distances; keep that semantic.
        dists.append(round(float(np.linalg.norm(pts - cam, axis=1).min()), 1))

    order = np.argsort(dists)
    best, second = order[0], order[1]
    flag = None
    if dists[best] < REL_DIST_MIN:
        flag = "closest_below_min_dist"
    elif abs(dists[best] - dists[second]) < REL_DIST_AMBIGUITY:
        flag = "ambiguous_after_fix"
    return categories[best], letters[best], flag


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test_json", required=True)
    ap.add_argument("--frame_meta", required=True)
    ap.add_argument("--scene_meta", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    with open(args.test_json) as f:
        data = json.load(f)
    with open(args.frame_meta) as f:
        frame_meta = json.load(f)
    with open(args.scene_meta) as f:
        scene_meta = json.load(f)

    if not HAS_O3D:
        print("NOTE: open3d not available; using numpy crop emulation "
              "(worst-case 0.1 m rounding flicker on bbox-distance labels).")

    cache = SceneCache(frame_meta, scene_meta)
    fixers = {
        "camera_displacement": fix_displacement,
        "camera_obj_abs_dist": fix_abs_dist,
        "camera_obj_rel_dist_v1": fix_rel_dist,
        "camera_obj_rel_dist_v2": fix_rel_dist,
        "camera_obj_rel_dist_v3": fix_rel_dist,
    }

    stats = defaultdict(Counter)
    corrected, dropped = [], []
    for item in data:
        qtype = item["question_type"]
        if qtype not in fixers:
            corrected.append(item)
            continue
        st = stats[qtype]
        st["total"] += 1
        new_gt, new_mc, flag = fixers[qtype](item, cache)
        if flag is not None:
            st[f"dropped:{flag}"] += 1
            dropped.append({**item, "drop_reason": flag})
            continue
        out = dict(item)
        if qtype.startswith("camera_obj_rel_dist"):
            if item["mc_answer"] != new_mc:
                st["label_changed"] += 1
            out["ground_truth"] = new_gt
            out["mc_answer"] = new_mc
        else:
            if float(item["ground_truth"]) != new_gt:
                st["label_changed"] += 1
            out["ground_truth"] = str(new_gt)
        corrected.append(out)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "test_corrected.json"), "w") as f:
        json.dump(corrected, f, indent=1)
    with open(os.path.join(args.output_dir, "dropped_items.json"), "w") as f:
        json.dump(dropped, f, indent=1)

    print(f"kept {len(corrected)} items, dropped {len(dropped)}")
    for qtype, st in sorted(stats.items()):
        kept = st["total"] - sum(v for k, v in st.items() if k.startswith("dropped:"))
        print(f"\n{qtype}: total {st['total']}, kept {kept}, "
              f"labels changed {st['label_changed']}")
        for k, v in sorted(st.items()):
            if k.startswith("dropped:"):
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
