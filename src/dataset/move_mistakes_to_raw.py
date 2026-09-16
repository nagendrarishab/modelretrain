"""
    # see what would move, touches nothing
    python src/dataset/move_mistakes_to_raw.py --mistakes-file "logs/yolo26n_best 3_mistakes.txt"

    # actually move images + labels into raw/ and raw_labels/
    python src/dataset/move_mistakes_to_raw.py --mistakes-file "logs/yolo26n_best 3_mistakes.txt" --move
"""
import argparse
import filecmp
import shutil
from pathlib import Path

import yaml

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def resolve_collision(src_image_path, dest_image_path, dest_label_path):
    """If dest_image_path is free, return it and dest_label_path unchanged.
    If it's taken by an identical file, return (None, None) to signal "skip,
    already there". If it's taken by a *different* file, find a free
    "<stem>_2<ext>" (etc.) name so the incoming file isn't silently dropped."""
    if not dest_image_path.exists():
        return dest_image_path, dest_label_path

    if filecmp.cmp(src_image_path, dest_image_path, shallow=False):
        return None, None

    stem, suffix = dest_image_path.stem, dest_image_path.suffix
    counter = 2
    while True:
        candidate_image = dest_image_path.with_name(f"{stem}_{counter}{suffix}")
        candidate_label = dest_label_path.with_name(f"{stem}_{counter}.txt")
        if not candidate_image.exists() and not candidate_label.exists():
            return candidate_image, candidate_label
        counter += 1


def read_mistakes(mistakes_file):
    names = []
    seen = set()
    for line in Path(mistakes_file).read_text().splitlines():
        name = line.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def classify_destination(label_path, class_names):
    if not label_path.exists():
        return "background"
    class_ids = set()
    for line in label_path.read_text().splitlines():
        if line.strip():
            class_ids.add(int(line.split()[0]))
    if not class_ids:
        return "background"
    if len(class_ids) == 1:
        return class_names[next(iter(class_ids))]
    return "extra"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mistakes-file", required=True,
                         help="e.g. 'logs/yolo26n_best 3_mistakes.txt' from evaluate_models.py")
    parser.add_argument("--data", default="testcase/data.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--raw-dir", default="raw")
    parser.add_argument("--raw-labels-dir", default="raw_labels")
    parser.add_argument("--move", action="store_true",
                         help="actually move files - default is a dry-run that only prints")
    args = parser.parse_args()

    data_yaml_path = Path(args.data)
    names_dict = yaml.safe_load(data_yaml_path.read_text())["names"]
    class_names = [names_dict[i] for i in sorted(names_dict)]

    split_dir = data_yaml_path.parent / args.split
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    raw_dir = Path(args.raw_dir)
    raw_labels_dir = Path(args.raw_labels_dir)

    image_names = read_mistakes(args.mistakes_file)
    print(f"{len(image_names)} image(s) listed in {args.mistakes_file}")

    counts = {}
    skipped = []

    for image_name in image_names:
        image_path = images_dir / image_name
        if not image_path.exists() and image_path.suffix.lower() not in IMAGE_EXTS:
            image_path = next((p for p in images_dir.glob(f"{Path(image_name).stem}.*")
                                if p.suffix.lower() in IMAGE_EXTS), image_path)
        if not image_path.exists():
            skipped.append((image_name, "image not found in " + str(images_dir)))
            continue

        label_path = labels_dir / f"{image_path.stem}.txt"
        dest_name = classify_destination(label_path, class_names)

        dest_image_path = raw_dir / dest_name / image_path.name
        dest_label_path = raw_labels_dir / dest_name / f"{image_path.stem}.txt"

        dest_image_path, dest_label_path = resolve_collision(image_path, dest_image_path, dest_label_path)
        if dest_image_path is None:
            skipped.append((image_name, f"identical file already at raw/{dest_name}/{image_path.name} - duplicate"))
            continue
        if dest_image_path.name != image_path.name:
            print(f"  note: {image_name} collides with a different existing file - "
                  f"renaming to {dest_image_path.name}")

        counts[dest_name] = counts.get(dest_name, 0) + 1

        if args.move:
            dest_image_path.parent.mkdir(parents=True, exist_ok=True)
            dest_label_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(image_path), str(dest_image_path))
            if label_path.exists():
                shutil.move(str(label_path), str(dest_label_path))
            else:
                dest_label_path.write_text("")
        else:
            print(f"  {image_path.name} -> {dest_image_path} (+ {dest_label_path.name})")

    print("\nBy destination folder:")
    for dest_name in sorted(counts):
        print(f"  {dest_name}: {counts[dest_name]}")

    if skipped:
        print(f"\nSkipped {len(skipped)} file(s):")
        for image_name, reason in skipped:
            print(f"  {image_name}: {reason}")

    if not args.move:
        print("\nDry-run only - nothing moved. Re-run with --move to actually move these files.")


if __name__ == "__main__":
    main()
