"""Create a deterministic, zero-copy dataset subset for CATK experiments."""

import argparse
import os
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Directory containing cached scenarios")
    parser.add_argument("output", type=Path, help="Empty directory to create")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=817)
    parser.add_argument(
        "--method",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="Use symlink to avoid duplicating cached scenarios",
    )
    return parser.parse_args()


def link_file(source: Path, destination: Path, method: str) -> None:
    if method == "symlink":
        destination.symlink_to(source.resolve())
    elif method == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()

    if not source.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source}")

    files = sorted(path for path in source.iterdir() if path.is_file())
    if args.count <= 0 or args.count > len(files):
        raise ValueError(f"count must be in [1, {len(files)}], got {args.count}")

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    selected = sorted(rng.sample(files, args.count), key=lambda path: path.name)
    for source_file in selected:
        link_file(source_file, output / source_file.name, args.method)

    manifest = output.parent / f"{output.name}.manifest.txt"
    manifest.write_text(
        "".join(f"{path.name}\n" for path in selected), encoding="utf-8"
    )
    print(f"Created {len(selected)} entries in {output}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
