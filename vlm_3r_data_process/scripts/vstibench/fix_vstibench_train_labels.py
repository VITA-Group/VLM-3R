"""Recompute VSTI training-QA answers affected by the camera-position bug.

Same correction as fix_vstibench_gt_labels.py, applied to the released
LLaVA-format training files (vstibench_train/qa_*.json in
Journey9ni/VLM-3R-DATA): question text lives in conversations[0], the answer
in conversations[1]. Labels are corrected in place so the training mix stays
as close as possible to the original; items whose corrected geometry violates
the generators' validity filters are dropped.

As a data-provenance check, the script also reports how many original answers
are reproduced by the buggy -R.T @ t camera position (expected ~100%).

Usage:
    python fix_vstibench_train_labels.py \
        --train_qa_dir old_train_qa/ \
        --frame_meta scannet_frame_metadata_train.json \
        --scene_meta scannet_metadata_train.json \
        --output_dir corrected_train_qa/
"""

import argparse
import json
import os
import re
from collections import Counter

import numpy as np
from tqdm import tqdm

from fix_vstibench_gt_labels import (
    SceneCache, DISPLACEMENT_RANGE, ABS_DIST_RANGE,
    REL_DIST_MIN, REL_DIST_AMBIGUITY,
    DISPLACEMENT_PAT, ABS_DIST_PAT, REL_DIST_PAT,
)

OPTION_LINE_PAT = re.compile(r"^([A-D])\.\s*(.+)$", re.MULTILINE)

AFFECTED_FILES = [
    "qa_camera_displacement",
    "qa_camera_obj_abs_dist",
    "qa_camera_obj_rel_dist_v1",
    "qa_camera_obj_rel_dist_v2",
    "qa_camera_obj_rel_dist_v3",
]


def buggy_camera_center(pose_list):
    P = np.asarray(pose_list, dtype=np.float64)
    return -P[:3, :3].T @ P[:3, 3]


def correct_camera_center(pose_list):
    P = np.asarray(pose_list, dtype=np.float64)
    return P[:3, 3]


def fix_item(item, cache):
    """Returns (new_answer or None, drop_reason or None, buggy_matches_old)."""
    qtype = item["question_type"]
    question = item["conversations"][0]["value"]
    old_answer = item["conversations"][1]["value"].strip()
    scene = item["scene_name"]

    if scene not in cache.frame_meta:
        return None, "scene_missing_in_metadata", False
    valid = cache.valid_frames(scene)

    if qtype == "camera_displacement":
        m = DISPLACEMENT_PAT.search(question)
        if not m:
            return None, "unparseable_question", False
        i, j, _ = map(int, m.groups())
        if j > len(valid):
            return None, "frame_rank_out_of_bounds", False
        p0, p1 = valid[i - 1], valid[j - 1]
        new = round(float(np.linalg.norm(
            correct_camera_center(p1["camera_pose_camera_to_world"]) -
            correct_camera_center(p0["camera_pose_camera_to_world"]))), 1)
        buggy = round(float(np.linalg.norm(
            buggy_camera_center(p1["camera_pose_camera_to_world"]) -
            buggy_camera_center(p0["camera_pose_camera_to_world"]))), 1)
        buggy_ok = abs(buggy - float(old_answer)) < 0.051
        if not (DISPLACEMENT_RANGE[0] <= new <= DISPLACEMENT_RANGE[1]):
            return None, "out_of_range", buggy_ok
        return str(new), None, buggy_ok

    if qtype == "camera_obj_abs_dist":
        m = ABS_DIST_PAT.search(question)
        if not m:
            return None, "unparseable_question", False
        category, k = m.group(1), int(m.group(2))
        if k > len(valid):
            return None, "frame_rank_out_of_bounds", False
        frame = valid[k - 1]
        iid = cache.frame_unique_instance(scene, frame, category)
        if iid is None:
            return None, "object_not_unique_in_frame", False
        pts = cache.instance_points(scene, iid)
        cam = correct_camera_center(frame["camera_pose_camera_to_world"])
        new = round(float(np.linalg.norm(pts - cam, axis=1).min()), 1)
        bcam = buggy_camera_center(frame["camera_pose_camera_to_world"])
        buggy = round(float(np.linalg.norm(pts - bcam, axis=1).min()), 1)
        buggy_ok = abs(buggy - float(old_answer)) < 0.051
        if not (ABS_DIST_RANGE[0] <= new <= ABS_DIST_RANGE[1]):
            return None, "out_of_range", buggy_ok
        return str(new), None, buggy_ok

    # camera_obj_rel_dist_v1/v2/v3
    m = REL_DIST_PAT.search(question)
    if not m:
        return None, "unparseable_question", False
    k = int(m.group(1))
    if k > len(valid):
        return None, "frame_rank_out_of_bounds", False
    frame = valid[k - 1]
    opts = OPTION_LINE_PAT.findall(question)
    if len(opts) < 2:
        return None, "unparseable_options", False
    letters = [o[0] for o in opts]
    categories = [o[1].strip() for o in opts]
    dists = []
    for cat in categories:
        iid = cache.frame_unique_instance(scene, frame, cat)
        if iid is None:
            return None, "option_not_unique_in_frame", False
        pts = cache.instance_points(scene, iid)
        cam = correct_camera_center(frame["camera_pose_camera_to_world"])
        dists.append(round(float(np.linalg.norm(pts - cam, axis=1).min()), 1))
    bcam = buggy_camera_center(frame["camera_pose_camera_to_world"])
    bdists = [round(float(np.linalg.norm(
        cache.instance_points(scene, cache.frame_unique_instance(scene, frame, cat))
        - bcam, axis=1).min()), 1) for cat in categories]
    buggy_ok = letters[int(np.argmin(bdists))] == old_answer

    order = np.argsort(dists)
    if dists[order[0]] < REL_DIST_MIN:
        return None, "closest_below_min_dist", buggy_ok
    if abs(dists[order[0]] - dists[order[1]]) < REL_DIST_AMBIGUITY:
        return None, "ambiguous_after_fix", buggy_ok
    return letters[int(order[0])], None, buggy_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train_qa_dir", required=True)
    ap.add_argument("--frame_meta", required=True)
    ap.add_argument("--scene_meta", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    with open(args.frame_meta) as f:
        frame_meta = json.load(f)
    with open(args.scene_meta) as f:
        scene_meta = json.load(f)
    cache = SceneCache(frame_meta, scene_meta)

    os.makedirs(args.output_dir, exist_ok=True)
    all_dropped = []
    for name in AFFECTED_FILES:
        path = os.path.join(args.train_qa_dir, f"{name}.json")
        with open(path) as f:
            data = json.load(f)
        kept, stats = [], Counter()
        buggy_match = 0
        for item in tqdm(data, desc=name):
            new_answer, drop, buggy_ok = fix_item(item, cache)
            buggy_match += bool(buggy_ok)
            if drop is not None:
                stats[f"dropped:{drop}"] += 1
                all_dropped.append({"file": name, "id": item["id"],
                                    "scene_name": item["scene_name"],
                                    "drop_reason": drop})
                continue
            if new_answer != item["conversations"][1]["value"].strip():
                stats["label_changed"] += 1
            out = json.loads(json.dumps(item))
            out["conversations"][1]["value"] = new_answer
            kept.append(out)
        with open(os.path.join(args.output_dir, f"{name}.json"), "w") as f:
            json.dump(kept, f, indent=1)
        print(f"{name}: total {len(data)}, kept {len(kept)}, "
              f"changed {stats['label_changed']}, "
              f"buggy-formula reproduces old answer {buggy_match}/{len(data)} "
              f"({100 * buggy_match / max(len(data), 1):.1f}%)")
        for k, v in sorted(stats.items()):
            if k.startswith("dropped:"):
                print(f"  {k}: {v}")

    with open(os.path.join(args.output_dir, "dropped_items.json"), "w") as f:
        json.dump(all_dropped, f, indent=1)
    print(f"\ntotal dropped: {len(all_dropped)}")


if __name__ == "__main__":
    main()
