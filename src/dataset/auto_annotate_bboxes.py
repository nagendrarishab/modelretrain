"""
Two ways to place a box, switchable per image with 'm':
  corners mode (default)  click 4 corners yourself - most precise, good for
                          an angled or partly occluded box
  SAM mode                click once on the object and MobileSAM (ultralytics'
                          SAM("mobile_sam.pt"), auto-downloaded on first use)
                          segments it and adds its mask's bounding rectangle -
                          usually a better fit than a hand-drawn box, and
                          faster when the box is clean and unoccluded. A bad
                          SAM box can be undone with 'r' and retried, or you
                          can switch back to corners mode ('m') and draw it
                          by hand instead.

Controls:
  left-click               corners mode: place one of a box's 4 corners (dots
                           connect as you click; the box - the axis-aligned
                           rectangle enclosing those 4 points - commits on the
                           4th click). SAM mode: query MobileSAM at that point
                           and add its suggested box immediately (low-confidence
                           or near-whole-image masks are rejected - re-click
                           more precisely on the object)
  m                        toggle between corners mode and SAM mode for boxes
                           you add from here on (shown in the window title bar)
  right-click a box        (raw/extra/ only) toggle that box's class between
                           open/closed
  o / c                    (raw/extra/ only) set the class new boxes will get
                           when drawn - shown in the window title bar
  y / n / Enter            save all boxes currently drawn, move to next
  e                        confirm this image has no box - save an empty label
                           and move to next (only when no boxes are drawn)
  r                        undo the last placed corner, or if none are
                           pending, the most recently added box
  s                        skip this image (no label saved, move on)
  b                        go back to the previous image
  q / Esc                  quit (progress already saved is kept)

"""
import argparse
import shutil
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import SAM, YOLO

CLASSES = ["closed", "open"]
MAX_DISPLAY = 900
CLASS_COLORS = {0: (0, 140, 255), 1: (0, 220, 0)}


MIN_BOX_SIZE = 3


class BoxState:
    def __init__(self):
        self.pending_corners = []
        self.boxes = []


def corners_to_box(corners, class_id):
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return min(xs), min(ys), max(xs), max(ys), class_id


def axis_aligned_to_corners(x1, y1, x2, y2):
    return (x1, y1), (x2, y1), (x2, y2), (x1, y2)


def make_mouse_callback(state, disp_w, disp_h, get_class, mixed, mode=None,
                         orig_w=None, orig_h=None, get_sam_model=None, img_path=None,
                         min_conf=0.5, status=None):

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def callback(event, x, y, flags, param):
        x = clamp(x, 0, disp_w - 1)
        y = clamp(y, 0, disp_h - 1)
        if event == cv2.EVENT_LBUTTONDOWN:
            if mode is not None and mode[0] == "sam":
                fx, fy = x * orig_w / disp_w, y * orig_h / disp_h
                status["message"] = "predicting..."
                try:
                    result = predict_box_at_point(get_sam_model(), img_path, fx, fy, min_conf)
                except Exception as e:
                    traceback.print_exc()
                    status["message"] = f"MobileSAM error: {e}"
                    return
                if result is None:
                    status["message"] = "no confident mask at that point - click more precisely on the object"
                    return
                x1, y1, x2, y2, conf = result
                if (x2 - x1) * (y2 - y1) / (orig_w * orig_h) > 0.9:
                    status["message"] = "mask covers nearly the whole image - probably background, ignored"
                    return
                (dx1, dy1, dx2, dy2), = full_rects_to_display([(x1, y1, x2, y2)], orig_w, orig_h, disp_w, disp_h)
                if dx2 - dx1 >= MIN_BOX_SIZE and dy2 - dy1 >= MIN_BOX_SIZE:
                    state.boxes.append((axis_aligned_to_corners(dx1, dy1, dx2, dy2), get_class()))
                    status["message"] = f"box added (conf {conf:.2f})"
                else:
                    status["message"] = "mask too small, ignored"
                return
            state.pending_corners.append((x, y))
            if len(state.pending_corners) == 4:
                corners = tuple(state.pending_corners)
                x1, y1, x2, y2, cid = corners_to_box(corners, get_class())
                state.pending_corners = []
                if x2 - x1 >= MIN_BOX_SIZE and y2 - y1 >= MIN_BOX_SIZE:
                    state.boxes.append((corners, cid))
        elif event == cv2.EVENT_RBUTTONDOWN and mixed:
            for i in range(len(state.boxes) - 1, -1, -1):  # topmost (last-drawn) box first
                corners, cid = state.boxes[i]
                contour = np.array(corners, dtype=np.int32)
                if cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0:
                    state.boxes[i] = (corners, 1 - cid)
                    break

    return callback


def rect_to_yolo_line(box, disp_w, disp_h):
    x1, y1, x2, y2, class_id = box
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    cx = (x1 + x2) / 2 / disp_w
    cy = (y1 + y2) / 2 / disp_h
    w = (x2 - x1) / disp_w
    h = (y2 - y1) / disp_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"


def load_full_image(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    return img, img.width, img.height


def query_generic_detector(detector, image, conf):
    results = detector.predict(image, conf=conf, verbose=False)[0]
    return [tuple(box) for box in results.boxes.xyxy.tolist()]


def predict_box_at_point(sam_model, img_path, x, y, min_conf):
    results = sam_model.predict(str(img_path), points=[[x, y]], labels=[1], verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    conf = float(boxes.conf[0])
    if conf < min_conf:
        return None
    x1, y1, x2, y2 = boxes.xyxy[0].tolist()
    return x1, y1, x2, y2, conf


def load_yolo_label(label_path, disp_w, disp_h):
    if not label_path.exists():
        return []
    rects = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        cid, cx, cy, w, h = line.split()
        cid = int(cid)
        cx, cy, w, h = float(cx), float(cy), float(w), float(h)
        x1 = (cx - w / 2) * disp_w
        y1 = (cy - h / 2) * disp_h
        x2 = (cx + w / 2) * disp_w
        y2 = (cy + h / 2) * disp_h
        rects.append((x1, y1, x2, y2, cid))
    return rects


def full_rects_to_display(rects, orig_w, orig_h, disp_w, disp_h):
    scale_x, scale_y = disp_w / orig_w, disp_h / orig_h
    clamp_x = lambda v: max(0, min(disp_w - 1, round(v * scale_x)))
    clamp_y = lambda v: max(0, min(disp_h - 1, round(v * scale_y)))
    return [(clamp_x(x1), clamp_y(y1), clamp_x(x2), clamp_y(y2)) for x1, y1, x2, y2 in rects]


def suggest_boxes(full_img, orig_w, orig_h, disp_w, disp_h, class_id, args, detector):
    default_class = class_id if class_id is not None else 0

    full_rects = query_generic_detector(detector, full_img, args.detector_conf)
    source = "generic detector" if full_rects else None
    if not full_rects:
        print("    generic detector found nothing; draw manually.")

    initial_rects = [(x1, y1, x2, y2, default_class)
                      for x1, y1, x2, y2 in full_rects_to_display(full_rects, orig_w, orig_h, disp_w, disp_h)]
    return initial_rects, source


def annotate_one(base_img, disp_w, disp_h, initial_rects, source, path, class_id, window,
                  orig_w=None, orig_h=None, get_sam_model=None, min_conf=0.5, mode_state=None):
    """orig_w/orig_h/get_sam_model - pass all three to enable SAM mode (the
    'm' key); omitted by callers (auto_annotate_with_model.py) that don't need it,
    in which case 'm' does nothing and only corners mode is available.

    mode_state - a 1-item list ("corners" or "sam") shared across the whole
    session by the caller, so switching modes on one image carries over to
    the next instead of resetting every time; a caller that doesn't pass one
    gets a fresh "corners" start each image instead."""
    mixed = class_id is None
    pending = [0]
    sam_available = get_sam_model is not None
    mode = mode_state if mode_state is not None else (["corners"] if sam_available else None)
    sam_status = {"message": ""}

    state = BoxState()
    state.boxes.extend((axis_aligned_to_corners(x1, y1, x2, y2), cid) for x1, y1, x2, y2, cid in initial_rects)
    cv2.setMouseCallback(window, make_mouse_callback(
        state, disp_w, disp_h, lambda: class_id if not mixed else pending[0], mixed,
        mode=mode, orig_w=orig_w, orig_h=orig_h, get_sam_model=get_sam_model,
        img_path=path, min_conf=min_conf, status=sam_status))

    while True:
        frame = base_img.copy()
        for corners, cid in state.boxes:
            color = CLASS_COLORS[cid]
            cv2.polylines(frame, [np.array(corners, dtype=np.int32)], isClosed=True, color=color, thickness=2)
            if mixed:
                x1, y1 = min(p[0] for p in corners), min(p[1] for p in corners)
                cv2.putText(frame, CLASSES[cid], (x1 + 3, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        corner_color = (0, 200, 255)
        for i, (cx, cy) in enumerate(state.pending_corners):
            cv2.circle(frame, (cx, cy), 4, corner_color, -1)
            if i > 0:
                cv2.line(frame, state.pending_corners[i - 1], (cx, cy), corner_color, 1)
        suggestion_note = f" [{source}: {len(initial_rects)} suggested]" if initial_rects else " [no suggestion]"
        corner_note = f" - {len(state.pending_corners)}/4 corners placed" if state.pending_corners else ""
        click_hint = "click the object" if sam_available and mode[0] == "sam" else "click 4 corners"
        status = (f" - no boxes, e=confirm empty" if not state.boxes
                  else f" - {len(state.boxes)} box(es), {click_hint} to add another, r=undo last")
        label = "mixed" if mixed else CLASSES[class_id]
        mode_note = f" [pending class: {CLASSES[pending[0]]} (o/c to change, right-click box to toggle)]" if mixed else ""
        sam_mode_note = f" [mode: {mode[0]} (m to toggle)]" if sam_available else ""
        sam_msg_note = f" - {sam_status['message']}" if sam_status["message"] else ""
        cv2.putText(frame, f"{path.name} [{label}]{suggestion_note}{status}{corner_note}"
                            f"{mode_note}{sam_mode_note}{sam_msg_note}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window, frame)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("y"), ord("n"), 13):  # y, n, or Enter - all save+advance
            if not state.boxes:
                continue
            lines = "".join(rect_to_yolo_line(corners_to_box(corners, cid), disp_w, disp_h)
                            for corners, cid in state.boxes)
            return ("save", lines)
        if key == ord("e") and not state.boxes and not state.pending_corners:
            return ("save", "")
        if key == ord("r"):
            if state.pending_corners:
                state.pending_corners.pop()
            elif state.boxes:
                state.boxes.pop()
        elif sam_available and key == ord("m"):
            mode[0] = "sam" if mode[0] == "corners" else "corners"
            state.pending_corners = []
            sam_status["message"] = ""
        elif mixed and key == ord("o"):
            pending[0] = 1
        elif mixed and key == ord("c"):
            pending[0] = 0
        elif key == ord("s"):
            return ("skip", None)
        elif key == ord("b"):
            return ("back", None)
        elif key in (ord("q"), 27):  # q or Esc
            return ("quit", None)


def route_dest(payload):
    """Which of closed/open/extra/background an annotated image belongs in,
    from the class(es) present in its saved YOLO-format label lines (empty
    payload - no boxes, confirmed via 'e' - means background)."""
    classes_present = {int(line.split()[0]) for line in payload.splitlines()}
    if not classes_present:
        return "background"
    if classes_present == {0}:
        return "closed"
    if classes_present == {1}:
        return "open"
    return "extra"


def route_and_save(payload, path, raw_dir, labels_dir):
    """Move an annotated image from its (unsorted) input location into
    raw_dir/<closed|open|extra|background>, and write its label alongside
    it there. Returns (dest, dest_img_path, dest_label_path)."""
    dest = route_dest(payload)
    (raw_dir / dest).mkdir(parents=True, exist_ok=True)
    (labels_dir / dest).mkdir(parents=True, exist_ok=True)
    dest_img = raw_dir / dest / path.name
    dest_label = labels_dir / dest / (path.stem + ".txt")
    shutil.move(str(path), str(dest_img))
    dest_label.write_text(payload)
    return dest, dest_img, dest_label


def run_sorting_session(args, input_dir, raw_dir, labels_dir, detector, window, get_sam_model, mode_state):
    """--input-dir mode: images start in one unsorted folder with no known
    class: annotate every one in mixed mode, then route_and_save() moves it
    into the right raw/<class> folder based on what got drawn.

    If input_dir is itself one of the raw/<class> folders (e.g. raw/background),
    a confirmed-empty image gets routed right back where it started instead of
    moving out - so "done" is tracked via its label file under
    labels_dir/<input_dir.name> instead, same as presorted mode."""
    all_found = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.jpeg"))
    if args.overwrite:
        pending = all_found
    else:
        pending = [f for f in all_found if not (labels_dir / input_dir.name / (f.stem + ".txt")).exists()]
    total = len(pending)
    print(f"{total}/{len(all_found)} images to sort+annotate from {input_dir} "
          f"(already-labeled ones skipped; use --overwrite to redo).")

    if args.start_after:
        names = [f.name for f in pending]
        try:
            resume_at = names.index(args.start_after) + 1
        except ValueError:
            raise SystemExit(f"--start-after {args.start_after!r} not found among the pending images.")
        pending = pending[resume_at:]
        print(f"Resuming after {args.start_after}: {len(pending)} left.")

    if not pending:
        print("Nothing to do.")
        return

    cv2.namedWindow(window)

    # idx -> (dest_img, dest_label) for images already routed this session, so
    # 'b' (back) can undo the move and re-open them for re-annotation.
    moved = {}
    skipped_idxs = set()

    idx = 0
    saved, skipped = 0, 0
    while idx < len(pending):
        path = pending[idx]  # stable original (input_dir) location for this item

        if idx in moved:
            dest_img, dest_label = moved.pop(idx)
            shutil.move(str(dest_img), str(path))
            dest_label.unlink(missing_ok=True)
            saved -= 1
        if idx in skipped_idxs:
            skipped_idxs.discard(idx)
            skipped -= 1

        print(f"[{idx + 1}/{len(pending)}] {path.name}")

        full_img, orig_w, orig_h = load_full_image(path)
        scale = min(1.0, MAX_DISPLAY / max(orig_w, orig_h))
        disp_w, disp_h = max(1, round(orig_w * scale)), max(1, round(orig_h * scale))
        disp_img = full_img.resize((disp_w, disp_h), Image.LANCZOS)
        base_img = cv2.cvtColor(np.array(disp_img), cv2.COLOR_RGB2BGR)

        initial_rects, source = suggest_boxes(
            full_img, orig_w, orig_h, disp_w, disp_h, None, args, detector)

        action, payload = annotate_one(base_img, disp_w, disp_h, initial_rects, source, path, None, window,
                                        orig_w=orig_w, orig_h=orig_h, get_sam_model=get_sam_model,
                                        min_conf=args.min_conf, mode_state=mode_state)

        if action == "save":
            dest, dest_img, dest_label = route_and_save(payload, path, raw_dir, labels_dir)
            moved[idx] = (dest_img, dest_label)
            print(f"    -> {dest}/{path.name}")
            saved += 1
            idx += 1
        elif action == "skip":
            skipped_idxs.add(idx)
            skipped += 1
            idx += 1
        elif action == "back":
            idx = max(0, idx - 1)
        elif action == "quit":
            break

    cv2.destroyAllWindows()
    print(f"\nRouted {saved} images to raw/{{closed,open,extra,background}}, skipped {skipped}, "
          f"{len(pending) - saved - skipped} left for next run.")


def run_presorted_session(args, raw_dir, labels_dir, detector, window, get_sam_model, mode_state):
    """Original mode: images already live in raw/closed, raw/open, raw/extra
    - each is annotated in place (raw/extra/ in mixed mode) and only its
    label file is written, nothing gets moved."""
    items = []
    for class_id, cls in enumerate(CLASSES):
        files = sorted((raw_dir / cls).glob("*.jpg")) + sorted((raw_dir / cls).glob("*.jpeg"))
        for f in files:
            items.append((f, class_id, cls))
    # raw/extra/: photos with both an open and a closed box in frame - class_id
    # is decided per box during annotation instead of once for the whole image.
    extra_files = sorted((raw_dir / "extra").glob("*.jpg")) + sorted((raw_dir / "extra").glob("*.jpeg"))
    for f in extra_files:
        items.append((f, None, "extra"))

    if not args.overwrite:
        pending = [(f, cid, cls) for f, cid, cls in items
                   if not (labels_dir / cls / (f.stem + ".txt")).exists()]
    else:
        pending = items

    total = len(items)
    print(f"{len(pending)}/{total} images need annotation (already-labeled ones are skipped; use --overwrite to redo).")

    if args.start_after:
        names = [f.name for f, _, _ in pending]
        try:
            resume_at = names.index(args.start_after) + 1
        except ValueError:
            raise SystemExit(f"--start-after {args.start_after!r} not found among the pending images.")
        pending = pending[resume_at:]
        print(f"Resuming after {args.start_after}: {len(pending)} left.")

    if not pending:
        print("Nothing to do.")
        return

    cv2.namedWindow(window)

    idx = 0
    saved, skipped = 0, 0
    while idx < len(pending):
        path, class_id, cls = pending[idx]
        (labels_dir / cls).mkdir(parents=True, exist_ok=True)
        print(f"[{idx + 1}/{len(pending)}] {cls}/{path.name}")

        full_img, orig_w, orig_h = load_full_image(path)
        scale = min(1.0, MAX_DISPLAY / max(orig_w, orig_h))
        disp_w, disp_h = max(1, round(orig_w * scale)), max(1, round(orig_h * scale))
        disp_img = full_img.resize((disp_w, disp_h), Image.LANCZOS)
        base_img = cv2.cvtColor(np.array(disp_img), cv2.COLOR_RGB2BGR)

        label_path = labels_dir / cls / (path.stem + ".txt")
        initial_rects = load_yolo_label(label_path, disp_w, disp_h)
        if initial_rects:
            source = "existing label"
        else:
            initial_rects, source = suggest_boxes(
                full_img, orig_w, orig_h, disp_w, disp_h, class_id, args, detector)

        action, payload = annotate_one(base_img, disp_w, disp_h, initial_rects, source, path, class_id, window,
                                        orig_w=orig_w, orig_h=orig_h, get_sam_model=get_sam_model,
                                        min_conf=args.min_conf, mode_state=mode_state)

        if action == "save":
            (labels_dir / cls / (path.stem + ".txt")).write_text(payload)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="raw",
                         help="root containing (or, with --input-dir, that will receive) the "
                              "closed/open/extra/background subfolders")
    parser.add_argument("--labels-dir", default="raw_labels")
    parser.add_argument("--input-dir", default=None,
                         help="a folder of unsorted images to annotate+sort instead of the "
                              "presorted raw/closed, raw/open, raw/extra - see the module "
                              "docstring. --overwrite doesn't apply in this mode: an image is "
                              "'done' once it's been moved out of here.")
    parser.add_argument("--detector-model", default="models/yolo26n_best.pt",
                         help="generic pretrained detector used to suggest boxes")
    parser.add_argument("--detector-conf", type=float, default=0.25,
                         help="confidence threshold for the fallback detector")
    parser.add_argument("--sam-model", default="mobile_sam.pt",
                         help="ultralytics SAM checkpoint for SAM mode ('m' key) - lazily loaded "
                              "(auto-downloaded on first use if not present) only if you actually "
                              "switch into SAM mode")
    parser.add_argument("--min-conf", type=float, default=0.5,
                         help="minimum MobileSAM mask confidence to accept a SAM-mode click as a box")
    parser.add_argument("--overwrite", action="store_true",
                         help="re-annotate images that already have a label - in --input-dir mode, "
                              "an image counts as already-labeled if labels_dir/<input-dir-name>/ "
                              "has a matching .txt file")
    parser.add_argument("--start-after", default=None,
                         help="resume a pass: skip images up to and including this filename "
                              "(e.g. IMG-20260811-WA0002.jpg), which is the last one you finished last run")
    args = parser.parse_args()

    detector = YOLO(args.detector_model)

    # SAM mode ('m' key) loads MobileSAM lazily - only the first time it's
    # actually switched into - so a session that never uses it pays no cost.
    sam_holder = {"model": None}

    def get_sam_model():
        if sam_holder["model"] is None:
            print(f"    loading MobileSAM ({args.sam_model})...")
            sam_holder["model"] = SAM(args.sam_model)
        return sam_holder["model"]

    # Shared across every image in this run: switching into SAM mode with 'm'
    # on one image carries forward to the next instead of resetting each time.
    mode_state = ["corners"]

    raw_dir = Path(args.raw_dir)
    labels_dir = Path(args.labels_dir)
    window = "Auto-annotate (click 4 corners=add box, m=toggle SAM mode, n/y/Enter=save+next, " \
             "e=confirm empty, r=undo last, s=skip, b=back, q=quit)"

    if args.input_dir:
        run_sorting_session(args, Path(args.input_dir), raw_dir, labels_dir, detector, window,
                             get_sam_model, mode_state)
    else:
        run_presorted_session(args, raw_dir, labels_dir, detector, window, get_sam_model, mode_state)


if __name__ == "__main__":
    main()
