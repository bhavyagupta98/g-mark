from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_coop_drive.infrastructure.v2vgot_benchmark_adapter import V2VGoTQABenchmarkAdapter

DEFAULT_CANDIDATES = (
    Path("/workspace/repos/V2V-GoT"),
    REPO_ROOT.parent / "V2V-GoT",
)


def resolve_v2vgot_root() -> Path:
    """Resolve the local V2V-GoT root for either pod or local development."""

    for candidate in DEFAULT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate the V2V-GoT repository. Set up the repo under "
        "/workspace/repos/V2V-GoT or as a sibling of kg_coop_drive."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect V2V-GoT-QA task categories.")
    parser.add_argument("--split", default="val", choices=("val", "train"))
    parser.add_argument(
        "--file-name",
        default="v2v4real_3d_grounding_qa_dataset_v2vgot.json",
    )
    parser.add_argument("--examples", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repository_root = resolve_v2vgot_root()
    adapter = V2VGoTQABenchmarkAdapter(str(repository_root))
    samples = adapter.load_samples(split_name=args.split, file_name=args.file_name)
    counts = adapter.summarize_task_inventory(split_name=args.split, file_name=args.file_name)

    print("=" * 72)
    print("V2V-GoT-QA Task Inventory")
    print("=" * 72)
    print(f"repository_root: {repository_root}")
    print(f"split: {args.split}")
    print(f"file_name: {args.file_name}")
    print(f"sample_count: {len(samples)}")
    print()
    print("Task Counts")
    print("-" * 72)
    for task_type, count in counts.items():
        print(f"- {task_type.value}: {count}")

    print()
    print("Example Questions")
    print("-" * 72)
    per_task_seen: dict[str, int] = {}
    for sample in samples:
        key = sample.task_type.value
        seen = per_task_seen.get(key, 0)
        if seen >= args.examples:
            continue
        print(f"[{key}] sample_id={sample.sample_id} qa_type_id={sample.qa_type_id}")
        print(f"question: {sample.scene.raw_question}")
        print()
        per_task_seen[key] = seen + 1


if __name__ == "__main__":
    main()
