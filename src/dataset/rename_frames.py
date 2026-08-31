"""
    Usage:
        python src/dataset/rename_frames.py --dir . --start 7000
        python src/dataset/rename_frames.py --dir . --continue-dir /path/to/existing_dataset
        python src/dataset/rename_frames.py --dir . --template "{n:05d}"

    Pass --labels-dir to also rename each image's matching YOLO-format label
    (same stem, a .txt in --labels-dir) so images and labels stay paired:
    python src/dataset/rename_frames.py --dir data_detect/train/images --labels-dir data_detect/train/labels --start 7000

Rename multiple files in a single time with continuation
python src/dataset/rename_frames.py --dir raw/closed --labels-dir raw_labels/closed --template "{n:07d}"
python src/dataset/rename_frames.py --dir raw/open --labels-dir raw_labels/open --continue-dir raw/closed --template "{n:07d}"
python src/dataset/rename_frames.py --dir raw/extra --labels-dir raw_labels/extra --continue-dir raw/open --template "{n:07d}"
python src/dataset/rename_frames.py --dir raw/background --continue-dir raw/extra --template "{n:07d}"

"""
import argparse
import re
from pathlib import Path

NUM_RE = re.compile(r"(\d+)")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}


def last_number_match(stem):
    matches = list(NUM_RE.finditer(stem))
    return matches[-1] if matches else None


def sort_key(path):
    match = last_number_match(path.stem)
    if match:
        return (0, int(match.group(1)), path.stem)
    return (1, 0, path.stem)


def list_images(dir_path, pattern):
    return [f for f in dir_path.glob(pattern) if f.is_file() and f.suffix.lower() in IMAGE_EXTS]


def find_last_number(dir_path, pattern):
    dir_path = Path(dir_path)
    numbers = []
    for f in list_images(dir_path, pattern):
        match = last_number_match(f.stem)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers) if numbers else None


def renumber(dir_path, pattern, start, width, template, labels_dir=None):
    dir_path = Path(dir_path)
    labels_dir = Path(labels_dir) if labels_dir else None
    if labels_dir and not labels_dir.is_dir():
        raise FileNotFoundError(f"No such directory: {labels_dir}")

    files = sorted(list_images(dir_path, pattern), key=sort_key)
    if not files:
        print(f"No files matching '{pattern}' found in {dir_path}")
        return

    if not template:
        match = last_number_match(files[0].stem)
        prefix = files[0].stem[:match.start()] if match else files[0].stem
        stem_suffix = files[0].stem[match.end():] if match else ""
        suffix = stem_suffix + files[0].suffix

    def build_name(n, ext):
        if template:
            return template.format(n=n) + ext
        return f"{prefix}{n:0{width}d}{suffix}"

    def build_stem(n):
        if template:
            return template.format(n=n)
        return f"{prefix}{n:0{width}d}{stem_suffix}"

    # Pair each image with its label (if any) before any renaming starts,
    # since the image's own stem is about to change.
    labels = [labels_dir / f"{f.stem}.txt" if labels_dir else None for f in files]
    labels = [p if p and p.exists() else None for p in labels]
    missing_labels = [f.name for f, label in zip(files, labels) if labels_dir and label is None]

    temp_images, temp_labels = [], []
    for f, label in zip(files, labels):
        temp_image = f.with_name(f".tmp_rename_{f.name}")
        f.rename(temp_image)
        temp_images.append(temp_image)

        temp_label = None
        if label:
            temp_label = label.with_name(f".tmp_rename_{label.name}")
            label.rename(temp_label)
        temp_labels.append(temp_label)

    for i, (f, temp_image, temp_label) in enumerate(zip(files, temp_images, temp_labels)):
        n = start + i
        temp_image.rename(dir_path / build_name(n, f.suffix))
        if temp_label:
            temp_label.rename(labels_dir / f"{build_stem(n)}.txt")

    last_n = start + len(files) - 1
    print(f"Renamed {len(files)} files to {build_name(start, files[0].suffix)} "
          f"through {build_name(last_n, files[-1].suffix)}")
    if labels_dir:
        renamed_labels = sum(1 for label in temp_labels if label)
        print(f"Renamed {renamed_labels} matching label(s) in {labels_dir}")
        if missing_labels:
            print(f"{len(missing_labels)} image(s) had no matching label, left as-is:")
            for name in missing_labels:
                print(f"  {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".", help="folder containing the files to renumber")
    parser.add_argument("--pattern", default="*", help="glob pattern to match (only image files are picked up regardless of extension)")
    parser.add_argument("--start", type=int, default=0, help="first index to renumber from (ignored if --continue-dir is set)")
    parser.add_argument("--width", type=int, default=5, help="zero-padding width for the number (ignored if --template is set)")
    parser.add_argument("--continue-dir", help="look here for the highest existing frame number and continue numbering right after it, instead of --start")
    parser.add_argument("--continue-pattern", default="*", help="glob pattern used inside --continue-dir to find the last number")
    parser.add_argument("--template", help='fully custom filename pattern with a {n} placeholder, without the extension '
                                             '(the original file\'s extension is kept), e.g. "{n:05d}"')
    parser.add_argument("--labels-dir", help="also rename each image's matching YOLO-format label "
                                              "(same stem, a .txt in this directory) to keep the two in sync")
    args = parser.parse_args()

    start = args.start
    if args.continue_dir:
        last = find_last_number(args.continue_dir, args.continue_pattern)
        start = last + 1 if last is not None else args.start
        print(f"Continuing from {args.continue_dir}: last number found = {last}, starting at {start}")

    renumber(args.dir, args.pattern, start, args.width, args.template, args.labels_dir)
