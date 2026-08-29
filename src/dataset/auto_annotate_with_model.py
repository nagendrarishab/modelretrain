"""
python src/dataset/auto_annotate_with_model.py \
    --images-dir testcase/test/images --labels-dir testcase/test/labels \
    --model-path yolo26n_best.pt --data testcase/data.yaml

Controls: same as auto_annotate_bboxes.py - drag to add a box, right-click a box
to toggle its class, o/c to set the class new boxes get, y/n/Enter to save+advance,
e to confirm this image has no box (saves an empty label - only when no boxes are
drawn), r to undo last box, s to skip, b to go back, q/Esc to quit (progress
already saved is kept).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate_models import identify_and_load, IMAGE_EXTS

from auto_annotate_bboxes import (
    MAX_DISPLAY,
    annotate_one,
    full_rects_to_display,
    load_full_image,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default="testcase/test/images")
    parser.add_argument("--labels-dir", default="testcase/test/labels")
    parser.add_argument("--model-path", required=True,
                         help="trained checkpoint used to suggest boxes+classes - any format "
                              "evaluate_models.py's identify_and_load() recognizes")
    parser.add_argument("--data", default="testcase/data.yaml")
    parser.add_argument("--conf", type=float, default=0.25,
                         help="lower than evaluate_models.py's eval default so borderline "
                              "boxes are still suggested for a human to confirm/reject")
    parser.add_argument("--overwrite", action="store_true", help="re-annotate images that already have a label")
    parser.add_argument("--start-after", default=None,
                         help="resume an --overwrite pass: skip images up to and including this "
                              "filename, the last one you finished last run")
    args = parser.parse_args()

    _, class_names, predictor = identify_and_load(args.model_path, args.conf, args.data)
    name_to_idx = {name: i for i, name in enumerate(class_names)}

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    items = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not args.overwrite:
        pending = [p for p in items if not (labels_dir / f"{p.stem}.txt").exists()]
    else:
        pending = items

    total = len(items)
    print(f"{len(pending)}/{total} images need annotation (already-labeled ones are skipped; use --overwrite to redo).")

    if args.start_after:
        names = [p.name for p in pending]
        try:
            resume_at = names.index(args.start_after) + 1
        except ValueError:
            raise SystemExit(f"--start-after {args.start_after!r} not found among the pending images.")
        pending = pending[resume_at:]
        print(f"Resuming after {args.start_after}: {len(pending)} left.")

    if not pending:
        print("Nothing to do.")
        return

    window = "Auto-annotate with model (drag=add box, right-click=toggle class, " \
             "n/y/Enter=save+next, e=confirm empty, r=undo last box, s=skip, b=back, q=quit)"
    cv2.namedWindow(window)

    idx = 0
    saved, skipped = 0, 0
    while idx < len(pending):
        path = pending[idx]
        print(f"[{idx + 1}/{len(pending)}] {path.name}")

        full_img, orig_w, orig_h = load_full_image(path)
        scale = min(1.0, MAX_DISPLAY / max(orig_w, orig_h))
        disp_w, disp_h = max(1, round(orig_w * scale)), max(1, round(orig_h * scale))
        disp_img = full_img.resize((disp_w, disp_h), Image.LANCZOS)
        base_img = cv2.cvtColor(np.array(disp_img), cv2.COLOR_RGB2BGR)

        frame_bgr = cv2.cvtColor(np.array(full_img), cv2.COLOR_RGB2BGR)
        preds = predictor(frame_bgr)
        full_rects = [(x1, y1, x2, y2) for x1, y1, x2, y2, _, _ in preds]
        pred_classes = [name_to_idx[cls_name] for *_, cls_name, _ in preds]
        disp_rects = full_rects_to_display(full_rects, orig_w, orig_h, disp_w, disp_h)
        initial_rects = [(x1, y1, x2, y2, cid) for (x1, y1, x2, y2), cid in zip(disp_rects, pred_classes)]
        source = "model" if initial_rects else None

        action, payload = annotate_one(base_img, disp_w, disp_h, initial_rects, source, path, None, window)

        if action == "save":
            (labels_dir / f"{path.stem}.txt").write_text(payload)
            saved += 1
            idx += 1
        elif action == "skip":
            skipped += 1
            idx += 1
        elif action == "back":
            idx = max(0, idx - 1)
        elif action == "quit":
            break

    cv2.destroyAllWindows()
    print(f"\nSaved {saved} labels, skipped {skipped}, {len(pending) - saved - skipped} left for next run.")


if __name__ == "__main__":
    main()
