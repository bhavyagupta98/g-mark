#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.infrastructure.v2vgot_processed_assets import V2VGoTProcessedAssetLoader


DEFAULT_V2VGOT_ROOTS = (
    Path("/workspace/repos/V2V-GoT"),
    REPO_ROOT.parent / "V2V-GoT",
)


def resolve_v2vgot_root() -> Path:
    env_value = os.environ.get("V2VGOT_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    for candidate in DEFAULT_V2VGOT_ROOTS:
        if candidate.exists():
            return candidate.resolve()
    return DEFAULT_V2VGOT_ROOTS[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit expected V2V-GoT processed assets.")
    parser.add_argument("--split", default="val", choices=("val", "train"))
    parser.add_argument("--timestamps", type=int, default=10, help="How many discovered timestamps to inspect.")
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Also scan every official_models/*/npy folder, not just the default loader roots.",
    )
    return parser.parse_args()


def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def summarize_root(root: Path, timestamps_to_inspect: int) -> dict[str, object]:
    npy_files = tuple(root.rglob("*.npy"))
    json_files = tuple(root.rglob("*.json"))
    timestamp_indices: set[int] = set()

    has_ego_dir = (root / "ego").exists()
    has_cav1_dir = (root / "1").exists()

    gt_boxes = 0
    gt_ids = 0
    pred_ego = 0
    pred_cav1 = 0
    vis_ego = 0
    vis_cav1 = 0

    for path in npy_files:
        stem = path.stem
        prefix = stem.split("_", 1)[0]
        if prefix.isdigit():
            timestamp_indices.add(int(prefix))

        if stem.endswith("_gt"):
            gt_boxes += 1
        elif stem.endswith("_gt_object_id"):
            gt_ids += 1
        elif stem.endswith("_gt_object_id_visible_to_ego") or stem.endswith("_gt_object_id_invisible_to_ego"):
            vis_ego += 1
        elif stem.endswith("_gt_object_id_visible_to_1") or stem.endswith("_gt_object_id_invisible_to_1"):
            vis_cav1 += 1

    if has_ego_dir:
        pred_ego = sum(1 for path in (root / "ego").glob("*_pred.npy"))
    else:
        pred_ego = sum(1 for path in root.glob("*_pred.npy"))

    if has_cav1_dir:
        pred_cav1 = sum(1 for path in (root / "1").glob("*_pred.npy"))
    else:
        pred_cav1 = 0

    inspected_timestamps = tuple(sorted(timestamp_indices))[:timestamps_to_inspect]

    return {
        "root": root,
        "npy_files": len(npy_files),
        "json_files": len(json_files),
        "timestamp_count": len(timestamp_indices),
        "sample_timestamps": inspected_timestamps,
        "gt_boxes_count": gt_boxes,
        "gt_ids_count": gt_ids,
        "pred_ego_count": pred_ego,
        "pred_cav1_count": pred_cav1,
        "visibility_ego_count": vis_ego,
        "visibility_cav1_count": vis_cav1,
        "has_ego_dir": has_ego_dir,
        "has_cav1_dir": has_cav1_dir,
    }


def main() -> None:
    args = parse_args()
    v2vgot_root = resolve_v2vgot_root()
    loader = V2VGoTProcessedAssetLoader(str(v2vgot_root))

    print_section("V2V-GoT Asset Audit")
    print(f"repository_root: {v2vgot_root}")
    print(f"split: {args.split}")

    print_section("Expected Archive Presence")
    archive_names = (
        "dataset_processed_features_and_gt.zip",
        "dataset_jsons.zip",
        "model_ckpt.zip",
        "DMSTrack/V2V4Real/official_models/no_fusion_keep_all.zip",
        "DMSTrack/V2V4Real/official_models/train_no_fusion_keep_all.zip",
        "DMSTrack/V2V4Real/official_models/cobevt.zip",
    )
    for name in archive_names:
        path = v2vgot_root / name
        print(f"- {name}: {'present' if path.exists() else 'missing'}")

    print_section("Candidate NPY Roots")
    candidate_roots = loader.list_candidate_npy_roots(split_name=args.split)
    if not candidate_roots:
        print("No candidate processed roots exist on disk.")
    for root in candidate_roots:
        npy_count = sum(1 for _ in root.rglob("*.npy"))
        json_count = sum(1 for _ in root.rglob("*.json"))
        print(f"- root={root}")
        print(f"  npy_files={npy_count}")
        print(f"  json_files={json_count}")

    if args.all_models:
        print_section("All official_models NPY Roots")
        official_models_root = v2vgot_root / "DMSTrack" / "V2V4Real" / "official_models"
        discovered_roots = sorted(
            model_root / "npy"
            for model_root in official_models_root.iterdir()
            if model_root.is_dir() and (model_root / "npy").exists()
        )
        if not discovered_roots:
            print("No official_models/*/npy folders found.")
        for root in discovered_roots:
            summary = summarize_root(root, args.timestamps)
            print(f"- model_root={summary['root']}")
            print(f"  npy_files={summary['npy_files']}")
            print(f"  json_files={summary['json_files']}")
            print(f"  timestamps={summary['timestamp_count']}")
            print(f"  sample_timestamps={list(summary['sample_timestamps'])}")
            print(f"  gt_boxes={summary['gt_boxes_count']}")
            print(f"  gt_ids={summary['gt_ids_count']}")
            print(f"  pred_ego={summary['pred_ego_count']}")
            print(f"  pred_cav1={summary['pred_cav1_count']}")
            print(f"  visibility_ego={summary['visibility_ego_count']}")
            print(f"  visibility_cav1={summary['visibility_cav1_count']}")
            print(f"  has_ego_dir={summary['has_ego_dir']}")
            print(f"  has_cav1_dir={summary['has_cav1_dir']}")

    timestamps = loader.list_available_timestamps(split_name=args.split)
    print_section("Discovered Timestamps")
    print(f"count: {len(timestamps)}")
    if timestamps:
        print(f"sample: {list(timestamps[:args.timestamps])}")
    else:
        print("No timestamps discovered from current processed roots.")

    inspected = 0
    gt_ok = 0
    both_pred_ok = 0
    print_section("Per-Timestamp Availability Sample")
    for timestamp in timestamps[: args.timestamps]:
        availability = loader.inspect_availability(timestamp_index=timestamp, split_name=args.split)
        print(
            f"- t={timestamp}: "
            f"gt_boxes={availability.has_gt_boxes}, "
            f"gt_ids={availability.has_gt_ids}, "
            f"pred_ego={availability.has_pred_for_ego}, "
            f"pred_cav1={availability.has_pred_for_cav1}, "
            f"vis_ego={availability.has_visibility_for_ego}, "
            f"vis_cav1={availability.has_visibility_for_cav1}, "
            f"root={availability.npy_root}"
        )
        inspected += 1
        if availability.has_gt_boxes and availability.has_gt_ids:
            gt_ok += 1
        if availability.has_pred_for_ego and availability.has_pred_for_cav1:
            both_pred_ok += 1

    print_section("Summary")
    print(f"timestamps_inspected: {inspected}")
    print(f"timestamps_with_gt: {gt_ok}")
    print(f"timestamps_with_both_agent_predictions: {both_pred_ok}")
    if inspected == 0:
        print("conclusion: processed frame assets are missing or undiscoverable in the current checkout.")
    elif both_pred_ok == 0:
        print("conclusion: processed roots exist, but no inspected timestamp exposes synchronized predictions for both agents.")
    else:
        print("conclusion: at least one inspected timestamp exposes synchronized predictions for both agents.")


if __name__ == "__main__":
    main()
