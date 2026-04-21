#!/usr/bin/env python3

from __future__ import annotations

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
    """Resolve the local V2V-GoT root for either pod or local development."""

    env_value = os.environ.get("V2VGOT_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()

    for candidate in DEFAULT_V2VGOT_ROOTS:
        if candidate.exists():
            return candidate.resolve()

    return DEFAULT_V2VGOT_ROOTS[0]


def main() -> None:
    repository_root = resolve_v2vgot_root()
    loader = V2VGoTProcessedAssetLoader(str(repository_root))

    print("Multi-agent prediction frame scan")
    print(f"repository_root: {repository_root}")
    candidate_roots = loader.list_candidate_npy_roots(split_name="val")
    if not candidate_roots:
        print("No processed roots currently exist on disk.")
        return

    any_matches = False
    for npy_root in candidate_roots:
        timestamps = loader.list_available_timestamps_for_root(npy_root)
        print()
        print(f"npy_root: {npy_root}")
        print(f"timestamps_scanned: {len(timestamps)}")
        if not timestamps:
            print("No processed timestamps were discovered under this root.")
            continue

        matches = []
        for timestamp_index in timestamps:
            has_pred_for_ego = loader.prediction_source_exists_for_root(
                npy_root=npy_root,
                timestamp_index=timestamp_index,
                cav_id="ego",
            )
            has_pred_for_cav1 = loader.prediction_source_exists_for_root(
                npy_root=npy_root,
                timestamp_index=timestamp_index,
                cav_id="1",
            )
            if has_pred_for_ego and has_pred_for_cav1:
                matches.append(timestamp_index)

        if not matches:
            print("No timestamps currently expose predictions for both CAV_EGO and CAV_1.")
            continue

        any_matches = True
        print(f"first_multi_agent_timestamp: {matches[0]}")
        print("matching_timestamps:")
        for timestamp_index in matches[:20]:
            print(f"- {timestamp_index}")

    if not any_matches:
        print()
        print("No candidate processed root currently exposes predictions for both CAV_EGO and CAV_1.")


if __name__ == "__main__":
    main()
