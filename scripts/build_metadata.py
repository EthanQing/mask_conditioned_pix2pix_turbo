from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build_metadata(root: Path, output: Path) -> int:
    rows: list[dict[str, str]] = []
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        target_dir = root / "targets" / split
        mask_dir = root / "masks" / split
        if not image_dir.exists():
            continue
        for source_path in sorted(image_dir.iterdir()):
            if source_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            sample_id = source_path.stem
            target_candidates = [target_dir / f"{sample_id}{ext}" for ext in (".jpg", ".jpeg", ".png", ".webp")]
            mask_candidates = [mask_dir / f"{sample_id}{ext}" for ext in (".png", ".jpg", ".jpeg", ".webp")]
            target_path = next((p for p in target_candidates if p.exists()), None)
            mask_path = next((p for p in mask_candidates if p.exists()), None)
            if target_path is None or mask_path is None:
                continue
            rows.append({
                "id": sample_id,
                "split": split,
                "source_path": source_path.relative_to(root).as_posix(),
                "target_path": target_path.relative_to(root).as_posix(),
                "mask_path": mask_path.relative_to(root).as_posix(),
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "split", "source_path", "target_path", "mask_path"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("dataset/metadata.csv"))
    args = parser.parse_args()
    count = build_metadata(args.root, args.output)
    print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
