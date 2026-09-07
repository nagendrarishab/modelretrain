"""
Requires an API key in a .env file at the project root for whichever
--provider is selected: GEMINI_API_KEY for gemini, OPENROUTER_API_KEY for
openrouter.

Goes straight to the generic detector fallback by default - no key needed,
and skips the network round-trip + retries entirely. Pass --no-skip-vlm to
query the VLM for tier-1 box suggestions instead (needs an API key above).

Supports multiple boxes per image: both the VLM and the generic detector
suggest every box they find (not just the top one), and each set of 4
corner clicks adds a new box rather than replacing the previous one, so if
more than one physical box is in frame, they're pre-loaded together - or
click 4 corners per box yourself. Clicking the 4 actual corners (rather
than dragging a single diagonal) gets a tighter box on a box photographed
at an angle - the saved label is still the axis-aligned rectangle enclosing
those 4 points, just placed more precisely.

Images in raw/extra/ are treated as "mixed": instead of one class for the
whole image, each box gets its own open/closed class - for photos where a
closed box and an open box both appear in frame together.

Pass --input-dir to annotate an unsorted folder instead: every image is
shown in mixed mode (no pre-known class), and once you save it, the image
itself is moved into raw/closed, raw/open, raw/extra, or raw/background -
whichever matches the class(es) of the box(es) you drew (no boxes -> confirm
empty with e -> background) - and its label is written alongside it. This
replaces manually pre-sorting a fresh batch of photos into those folders
before labeling them.

Controls:
  left-click x4             place a box's 4 corners, in any order - dots
                           connect as you click, and the box (the axis-
                           aligned rectangle enclosing those 4 points) commits
                           on the 4th click (suggested boxes, if any, are
                           pre-loaded first - click 4 more corners to add
                           another)
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
import base64
import io
import json
import os
import shutil
import time
from pathlib import Path

import cv2
import httpx
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageOps
from ultralytics import YOLO

CLASSES = ["closed", "open"]
MAX_DISPLAY = 900
CLASS_COLORS = {0: (0, 140, 255), 1: (0, 220, 0)}  # index matches CLASSES order


MIN_BOX_SIZE = 3  # display px - ignore 4 corners clicked too close together to be a real box


class BoxState:
    def __init__(self):
        self.pending_corners = []  # up to 4 clicked (x, y) points for the box being placed
        self.boxes = []  # committed (corners, class_id) - corners = the 4 points as clicked,
                          # in click order, display pixel coords (whatever quadrilateral that
                          # traces - not forced to axis-aligned; only the saved label is)


def corners_to_box(corners, class_id):
    """4 corner points -> the axis-aligned rectangle enclosing them (class-
    agnostic of click order). Used to derive the saved label and for the
    min-size check - not for on-screen rendering, which shows the actual
    quadrilateral instead."""
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return min(xs), min(ys), max(xs), max(ys), class_id


def axis_aligned_to_corners(x1, y1, x2, y2):
    """The reverse direction, for seeding VLM/detector-suggested boxes (which
    come in already axis-aligned) into the same corners-based storage."""
    return (x1, y1), (x2, y1), (x2, y2), (x1, y2)


def make_mouse_callback(state, disp_w, disp_h, get_class, mixed):
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def callback(event, x, y, flags, param):
        x = clamp(x, 0, disp_w - 1)
        y = clamp(y, 0, disp_h - 1)
        if event == cv2.EVENT_LBUTTONDOWN:
            state.pending_corners.append((x, y))
            if len(state.pending_corners) == 4:
                corners = tuple(state.pending_corners)
                x1, y1, x2, y2, cid = corners_to_box(corners, get_class())
                state.pending_corners = []
                if x2 - x1 >= MIN_BOX_SIZE and y2 - y1 >= MIN_BOX_SIZE:  # ignore accidental clicks
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


PROMPT = """Find every plastic storage box/container (each has a flat lid and may \
be open or closed) in this image - there may be more than one. Respond with ONLY a
JSON array, no other text.

Include one object per box found:
[{"box_2d": [ymin, xmin, ymax, xmax]}, ...]
where each coordinate is normalized to 0-1000 relative to image height/width.

If no such box is visible anywhere in the image, respond with an empty array: []
"""


def load_full_image(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    return img, img.width, img.height


def _parse_box_response(text, image):
    """Parse a model's raw text reply into a list of (x1, y1, x2, y2) pixel
    coords - empty if it reported no box. Shared by every VLM provider,
    since they're all prompted for the identical box_2d JSON format."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("[") :]
    boxes = json.loads(text)
    rects = []
    for b in boxes:
        ymin, xmin, ymax, xmax = b["box_2d"]
        x1 = xmin / 1000 * image.width
        y1 = ymin / 1000 * image.height
        x2 = xmax / 1000 * image.width
        y2 = ymax / 1000 * image.height
        rects.append((x1, y1, x2, y2))
    return rects


def query_gemini_box(client, model, image, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[image, PROMPT],
                config=types.GenerateContentConfig(temperature=0),
            )
            return _parse_box_response(response.text, image)
        except Exception as e:
            wait = 2**attempt
            print(f"    Gemini request failed ({e}); retrying in {wait}s..." if attempt + 1 < max_retries
                  else f"    Gemini request failed ({e}); giving up, leave box empty.")
            if attempt + 1 < max_retries:
                time.sleep(wait)
    return []


def query_openrouter_box(api_key, model, image, max_retries=3):
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    data_url = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

    for attempt in range(max_retries):
        try:
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]}],
                    "temperature": 0,
                },
                timeout=60,
            )
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"]
            return _parse_box_response(text, image)
        except Exception as e:
            wait = 2**attempt
            print(f"    OpenRouter request failed ({e}); retrying in {wait}s..." if attempt + 1 < max_retries
                  else f"    OpenRouter request failed ({e}); giving up, leave box empty.")
            if attempt + 1 < max_retries:
                time.sleep(wait)
    return []


def query_generic_detector(detector, image, conf):
    """Return every box the detector finds above conf (any class), in the
    image's own pixel coordinates."""
    results = detector.predict(image, conf=conf, verbose=False)[0]
    return [tuple(box) for box in results.boxes.xyxy.tolist()]


def load_yolo_label(label_path, disp_w, disp_h):
    """Read a previously saved YOLO-format label file and convert it back to
    (x1, y1, x2, y2, class_id) rects in display pixel coords - lets an
    already-labeled image be reopened with its saved boxes pre-loaded instead
    of re-running the VLM/detector. Returns [] if the file doesn't exist or
    has no boxes."""
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


def annotate_one(base_img, disp_w, disp_h, initial_rects, source, path, class_id, window):
    mixed = class_id is None
    pending = [0]  # class new boxes get in mixed mode; toggled with o/c

    state = BoxState()
    state.boxes.extend((axis_aligned_to_corners(x1, y1, x2, y2), cid) for x1, y1, x2, y2, cid in initial_rects)
    cv2.setMouseCallback(
        window, make_mouse_callback(state, disp_w, disp_h, lambda: class_id if not mixed else pending[0], mixed))

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
        status = (f" - no boxes, e=confirm empty" if not state.boxes
                  else f" - {len(state.boxes)} box(es), click 4 corners to add another, r=undo last")
        label = "mixed" if mixed else CLASSES[class_id]
        mode_note = f" [pending class: {CLASSES[pending[0]]} (o/c to change, right-click box to toggle)]" if mixed else ""
        cv2.putText(frame, f"{path.name} [{label}]{suggestion_note}{status}{corner_note}{mode_note}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window, frame)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("y"), ord("n"), 13):  # y, n, or Enter - all save+advance
            if not state.boxes:
                continue  # need at least one box before advancing
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


def run_sorting_session(args, input_dir, raw_dir, labels_dir, vlm_client, detector, window):
    """--input-dir mode: images start in one unsorted folder with no known
    class: annotate every one in mixed mode, then route_and_save() moves it
    into the right raw/<class> folder based on what got drawn."""
    pending = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.jpeg"))
    total = len(pending)
    print(f"{total} images to sort+annotate from {input_dir}.")

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

        full_rects, source = [], None
        if not args.skip_vlm:
            print(f"    querying {args.provider}...")
            if args.provider == "gemini":
                full_rects = query_gemini_box(vlm_client, args.model, full_img)
            else:
                full_rects = query_openrouter_box(vlm_client, args.openrouter_model, full_img)
            source = args.provider
            if not full_rects:
                print(f"    {args.provider} found nothing; trying generic detector...")
        if not full_rects:
            full_rects = query_generic_detector(detector, full_img, args.detector_conf)
            source = "generic detector"
        if not full_rects:
            print("    generic detector found nothing either; draw manually.")
            source = None
        initial_rects = [(x1, y1, x2, y2, 0)
                          for x1, y1, x2, y2 in full_rects_to_display(full_rects, orig_w, orig_h, disp_w, disp_h)]

        action, payload = annotate_one(base_img, disp_w, disp_h, initial_rects, source, path, None, window)

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


def run_presorted_session(args, raw_dir, labels_dir, vlm_client, detector, window):
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
            full_rects, source = [], None
            if not args.skip_vlm:
                print(f"    querying {args.provider}...")
                if args.provider == "gemini":
                    full_rects = query_gemini_box(vlm_client, args.model, full_img)
                else:
                    full_rects = query_openrouter_box(vlm_client, args.openrouter_model, full_img)
                source = args.provider
                if not full_rects:
                    print(f"    {args.provider} found nothing; trying generic detector...")
            if not full_rects:
                full_rects = query_generic_detector(detector, full_img, args.detector_conf)
                source = "generic detector"
            if not full_rects:
                print("    generic detector found nothing either; draw manually.")
                source = None
            default_class = class_id if class_id is not None else 0
            initial_rects = [(x1, y1, x2, y2, default_class)
                              for x1, y1, x2, y2 in full_rects_to_display(full_rects, orig_w, orig_h, disp_w, disp_h)]

        action, payload = annotate_one(base_img, disp_w, disp_h, initial_rects, source, path, class_id, window)

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
    parser.add_argument("--provider", choices=["gemini", "openrouter"], default="openrouter",
                         help="which VLM to query for tier-1 box suggestions")
    parser.add_argument("--model", default="gemini-3.6-flash", help="Gemini model name (only used with --provider gemini)")
    parser.add_argument("--openrouter-model", default="nvidia/nemotron-nano-12b-v2-vl:free",
                         help="OpenRouter model name (only used with --provider openrouter)")
    parser.add_argument("--detector-model", default="models/yolo26n_best.pt",
                         help="generic pretrained detector used as fallback when the VLM finds nothing")
    parser.add_argument("--detector-conf", type=float, default=0.25,
                         help="confidence threshold for the fallback detector")
    parser.add_argument("--overwrite", action="store_true",
                         help="re-annotate images that already have a label (presorted mode only)")
    parser.add_argument("--start-after", default=None,
                         help="resume a pass: skip images up to and including this filename "
                              "(e.g. IMG-20260811-WA0002.jpg), which is the last one you finished last run")
    parser.add_argument("--skip-vlm", action=argparse.BooleanOptionalAction, default=True,
                         help="go straight to the local detector fallback, e.g. while the VLM's free quota is "
                              "exhausted (default: on - pass --no-skip-vlm to query the VLM instead)")
    args = parser.parse_args()

    vlm_client = None
    if not args.skip_vlm:
        load_dotenv()
        env_var = "GEMINI_API_KEY" if args.provider == "gemini" else "OPENROUTER_API_KEY"
        api_key = os.environ.get(env_var)
        if not api_key:
            raise SystemExit(f"Set {env_var} in a .env file, or pass --skip-vlm to use only the local detector.")
        # query_gemini_box needs a genai.Client; query_openrouter_box just needs the raw key.
        vlm_client = genai.Client(api_key=api_key) if args.provider == "gemini" else api_key
    detector = YOLO(args.detector_model)

    raw_dir = Path(args.raw_dir)
    labels_dir = Path(args.labels_dir)
    window = "Auto-annotate (click 4 corners=add box, n/y/Enter=save+next, e=confirm empty, r=undo last, s=skip, b=back, q=quit)"

    if args.input_dir:
        run_sorting_session(args, Path(args.input_dir), raw_dir, labels_dir, vlm_client, detector, window)
    else:
        run_presorted_session(args, raw_dir, labels_dir, vlm_client, detector, window)


if __name__ == "__main__":
    main()
