"""
Find images that don't have a matching label file (same filename stem, a
.txt in --labels-dir) and delete them. Dry-run by default - prints what
would be deleted without touching anything; pass --delete to actually do it.

    # see what would be deleted, touches nothing
    python src/dataset/clean_unlabeled_images.py --images-dir raw/open --labels-dir raw_labels/open

    # actually delete them
    python src/dataset/clean_unlabeled_images.py --images-dir raw/open --labels-dir raw_labels/open --delete

Also reports label files with no matching image - usually copy/paste debris
rather than "forgot to annotate this one", but a different kind of problem,
so it's a separate --delete-orphan-labels flag rather than being covered by
--delete:

    python src/dataset/clean_unlabeled_images.py --images-dir raw/open --labels-dir raw_labels/open --delete-orphan-labels
"""
import argparse
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--delete", action="store_true",
                         help="actually delete unlabeled images; without this, only reports what would happen")
    parser.add_argument("--delete-orphan-labels", action="store_true",
                         help="also delete label files with no matching image; without this, they're only reported")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"No such directory: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"No such directory: {labels_dir}")

    images = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    label_stems = {p.stem for p in labels_dir.glob("*.txt")}
    image_stems = {p.stem for p in images}

    unlabeled_images = sorted(p for p in images if p.stem not in label_stems)
    orphan_labels = sorted(labels_dir.glob("*.txt"))
    orphan_labels = [p for p in orphan_labels if p.stem not in image_stems]

    print(f"{len(images)} images in {images_dir}, {len(label_stems)} labels in {labels_dir}")

    if orphan_labels:
        label = "Deleted" if args.delete_orphan_labels else "Found (not deleted, just flagged)"
        print(f"\n{label} {len(orphan_labels)} label file(s) with no matching image:")
        for p in orphan_labels:
            print(f"  {p}")
        if args.delete_orphan_labels:
            for p in orphan_labels:
                p.unlink()
        else:
            print("Re-run with --delete-orphan-labels to actually remove these.")

    if not unlabeled_images:
        print("\nNo unlabeled images found.")
        return

    print(f"\n{len(unlabeled_images)} image(s) with no matching label:")
    for p in unlabeled_images:
        print(f"  {p}")

    if args.delete:
        for p in unlabeled_images:
            p.unlink()
        print(f"\nDeleted {len(unlabeled_images)} unlabeled image(s).")
    else:
        print(f"\nDry run - nothing deleted. Re-run with --delete to actually remove these {len(unlabeled_images)} file(s).")


if __name__ == "__main__":
    main()
