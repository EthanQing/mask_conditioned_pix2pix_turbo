from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def split_image(path: Path, source_out: Path, target_out: Path, separator_width: int, overwrite: bool) -> str | None:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size
            usable_w = w - separator_width
            if usable_w <= 0 or usable_w % 2 != 0:
                return f"{path}: invalid width={w} separator_width={separator_width}"
            half = usable_w // 2
            left_box = (0, 0, half, h)
            right_start = half + separator_width
            right_box = (right_start, 0, right_start + half, h)
            source = img.crop(left_box)
            target = img.crop(right_box)
            if source.size != target.size:
                return f"{path}: split sizes differ source={source.size} target={target.size}"
            source_out.parent.mkdir(parents=True, exist_ok=True)
            target_out.parent.mkdir(parents=True, exist_ok=True)
            if not overwrite and (source_out.exists() or target_out.exists()):
                return None
            source.save(source_out)
            target.save(target_out)
            return None
    except Exception as exc:
        return f"{path}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--separator-width", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--error-log", type=Path, default=Path("outputs/split_errors.log"))
    args = parser.parse_args()

    files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    errors: list[str] = []
    for path in files:
        out_name = f"{path.stem}.jpg"
        err = split_image(
            path,
            args.dataset_root / "images" / args.split / out_name,
            args.dataset_root / "targets" / args.split / out_name,
            args.separator_width,
            args.overwrite,
        )
        if err:
            errors.append(err)
    if errors:
        args.error_log.parent.mkdir(parents=True, exist_ok=True)
        args.error_log.write_text("\n".join(errors), encoding="utf-8")
    print(f"Processed {len(files)} files, errors={len(errors)}")


if __name__ == "__main__":
    main()
