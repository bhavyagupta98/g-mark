#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.infrastructure.v2vgot_processed_assets import V2VGoTProcessedAssetLoader


def resolve_v2vgot_root() -> Path:
    env_value = os.environ.get("V2VGOT_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()

    candidates = (
        Path("/workspace/repos/V2V-GoT"),
        REPO_ROOT.parent / "V2V-GoT",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]


def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def run_command(args: list[str], cwd: Path | None = None) -> None:
    print(f"$ {' '.join(args)}")
    subprocess.run(args, cwd=str(cwd) if cwd else None, check=True)


def inspect_explicit_root(v2vgot_root: Path) -> None:
    loader = V2VGoTProcessedAssetLoader(str(v2vgot_root))
    target = v2vgot_root / "DMSTrack" / "V2V4Real" / "official_models" / "no_fusion_keep_all" / "npy"

    timestamps = loader.list_available_timestamps_for_root(target)
    print(f"target_root: {target}")
    print(f"timestamp_count: {len(timestamps)}")
    print(f"sample_timestamps: {list(timestamps[:10])}")

    for timestamp in timestamps[:10]:
        availability = loader.inspect_availability(timestamp_index=timestamp, split_name="val")
        print(
            f"t={timestamp}: "
            f"gt_boxes={availability.has_gt_boxes}, "
            f"gt_ids={availability.has_gt_ids}, "
            f"pred_ego={availability.has_pred_for_ego}, "
            f"pred_cav1={availability.has_pred_for_cav1}, "
            f"vis_ego={availability.has_visibility_for_ego}, "
            f"vis_cav1={availability.has_visibility_for_cav1}, "
            f"resolved_root={availability.npy_root}"
        )


def show_layout(v2vgot_root: Path) -> None:
    target = v2vgot_root / "DMSTrack" / "V2V4Real" / "official_models" / "no_fusion_keep_all" / "npy"
    print(f"layout_root: {target}")
    dirs = sorted(path for path in target.rglob("*") if path.is_dir())
    for path in [target, *dirs[:40]]:
        print(path)


def main() -> None:
    v2vgot_root = resolve_v2vgot_root()

    print_section("1. Full asset audit for val")
    run_command(
        ["python3", "scripts/audit_v2vgot_assets.py", "--split", "val", "--timestamps", "10", "--all-models"],
        cwd=REPO_ROOT,
    )

    print_section("2. Full asset audit for train")
    run_command(
        ["python3", "scripts/audit_v2vgot_assets.py", "--split", "train", "--timestamps", "10", "--all-models"],
        cwd=REPO_ROOT,
    )

    print_section("3. Explicit no_fusion_keep_all inspection")
    inspect_explicit_root(v2vgot_root)

    print_section("4. no_fusion_keep_all folder layout")
    show_layout(v2vgot_root)

    print_section("5. Scene query demo")
    run_command(["python3", "scripts/demo_scene_query.py"], cwd=REPO_ROOT)

    print_section("6. Local graph validation")
    run_command(
        ["python3", "scripts/validate_local_graphs.py", "--max-frames", "5"],
        cwd=REPO_ROOT,
    )


if __name__ == "__main__":
    main()
