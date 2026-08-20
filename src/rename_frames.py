"""
    Usage:
        python src/rename_frames.py --dir . --start 7000
"""
import argparse
import re
from pathlib import Path

NUM_RE = re.compile(r"(\d+)")


def sort_key(path):
    match = NUM_RE.search(path.stem)
    return int(match.group(1)) if match else path.stem


def renumber(dir_path, pattern, start, width):
    dir_path = Path(dir_path)
    files = sorted(dir_path.glob(pattern), key=sort_key)
    if not files:
        print(f"No files matching '{pattern}' found in {dir_path}")
        return

    prefix = NUM_RE.split(files[0].stem, maxsplit=1)[0]
    suffix = files[0].suffix

    temp_paths = []
    for f in files:
        temp_path = f.with_name(f".tmp_rename_{f.name}")
        f.rename(temp_path)
        temp_paths.append(temp_path)

    for i, temp_path in enumerate(temp_paths):
        new_name = f"{prefix}{start + i:0{width}d}{suffix}"
        temp_path.rename(dir_path / new_name)

    print(f"Renamed {len(files)} files to {prefix}{start:0{width}d}{suffix} "
          f"through {prefix}{start + len(files) - 1:0{width}d}{suffix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".", help="folder containing the files to renumber")
    parser.add_argument("--pattern", default="frame_*.jpg", help="glob pattern to match")
    parser.add_argument("--start", type=int, default=7000, help="first index to renumber from")
    parser.add_argument("--width", type=int, default=5, help="zero-padding width for the number")
    args = parser.parse_args()

    renumber(args.dir, args.pattern, args.start, args.width)
