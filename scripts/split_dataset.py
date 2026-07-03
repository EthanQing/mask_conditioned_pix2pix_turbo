from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
SPLITS = ("train", "val", "test")


def list_files_by_stem(directory: Path, exts: set[str]) -> dict[str, Path]:
    if not directory.exists():
        raise FileNotFoundError(directory)
    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in exts
    }


def compute_split_counts(total: int) -> dict[str, int]:
    test_count = round(total * 0.03)
    val_count = round(total * 0.07)
    train_count = total - val_count - test_count
    return {"train": train_count, "val": val_count, "test": test_count}


def copy_or_move(src: Path, dst: Path, move: bool, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not overwrite:
            raise FileExistsError(dst)
        dst.unlink()
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def write_metadata(root: Path, source_dir_name: str, rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "split", "source_path", "target_path", "mask_path"])
        writer.writeheader()
        for row in rows:
            sample_id = row["id"]
            split = row["split"]
            writer.writerow(
                {
                    "id": sample_id,
                    "split": split,
                    "source_path": f"{source_dir_name}/{split}/{row['source_name']}",
                    "target_path": f"targets/{split}/{row['target_name']}",
                    "mask_path": f"masks/{split}/{row['mask_name']}",
                }
            )


def split_dataset(
    root: Path,
    source_dir_name: str,
    seed: int,
    move: bool,
    overwrite: bool,
    metadata_output: Path | None,
) -> tuple[dict[str, int], list[str]]:
    source_files = list_files_by_stem(root / source_dir_name, IMAGE_EXTS)
    target_files = list_files_by_stem(root / "targets", IMAGE_EXTS)
    mask_files = list_files_by_stem(root / "masks", MASK_EXTS)

    sample_ids = sorted(set(source_files) & set(target_files) & set(mask_files))
    missing = sorted((set(source_files) | set(target_files) | set(mask_files)) - set(sample_ids))
    if not sample_ids:
        raise ValueError(f"No matched samples found under {root}")

    random.Random(seed).shuffle(sample_ids)
    counts = compute_split_counts(len(sample_ids))
    split_by_id: dict[str, str] = {}
    start = 0
    for split in SPLITS:
        end = start + counts[split]
        for sample_id in sample_ids[start:end]:
            split_by_id[sample_id] = split
        start = end

    metadata_rows: list[dict[str, str]] = []
    for sample_id in sample_ids:
        split = split_by_id[sample_id]
        source_path = source_files[sample_id]
        target_path = target_files[sample_id]
        mask_path = mask_files[sample_id]

        copy_or_move(source_path, root / source_dir_name / split / source_path.name, move, overwrite)
        copy_or_move(target_path, root / "targets" / split / target_path.name, move, overwrite)
        copy_or_move(mask_path, root / "masks" / split / mask_path.name, move, overwrite)
        metadata_rows.append(
            {
                "id": sample_id,
                "split": split,
                "source_name": source_path.name,
                "target_name": target_path.name,
                "mask_name": mask_path.name,
            }
        )

    if metadata_output is not None:
        write_metadata(root, source_dir_name, metadata_rows, metadata_output)

    return counts, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Split paired dataset files into train/val/test = 90%/7%/3%.")
    parser.add_argument("--root", type=Path, default=Path("dataset"))
    parser.add_argument("--source-dir", default=None, help="Source image directory name. Defaults to sources, or images if sources is absent.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--move", action="store_true", help="Move files instead of copying them.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files in split directories.")
    parser.add_argument("--no-metadata", action="store_true", help="Do not write metadata.csv.")
    args = parser.parse_args()

    source_dir = args.source_dir
    if source_dir is None:
        source_dir = "sources" if (args.root / "sources").exists() else "images"
    metadata_output = None if args.no_metadata else args.root / "metadata.csv"

    counts, missing = split_dataset(
        root=args.root,
        source_dir_name=source_dir,
        seed=args.seed,
        move=args.move,
        overwrite=args.overwrite,
        metadata_output=metadata_output,
    )
    print(
        f"Split complete: train={counts['train']} val={counts['val']} test={counts['test']} "
        f"(source_dir={source_dir}, seed={args.seed})"
    )
    if metadata_output is not None:
        print(f"Wrote metadata to {metadata_output}")
    if missing:
        print(f"Skipped {len(missing)} unmatched files: {', '.join(missing[:20])}")
        if len(missing) > 20:
            print("...")


if __name__ == "__main__":
    main()
