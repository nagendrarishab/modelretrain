"""
Requires an API key in a .env file at the project root for whichever
--provider is selected: GEMINI_API_KEY for gemini, OPENROUTER_API_KEY for
openrouter.

Pass --skip-vlm to go straight to the generic detector fallback - no key
needed, and skips the network round-trip + retries entirely. Useful while
the VLM provider's free-tier quota is exhausted (manual work)

Supports multiple boxes per image: both the VLM and the generic detector
suggest every box they find (not just the top one), and each mouse drag
adds a new box rather than replacing the previous one, so if more than one
physical box is in frame, they're pre-loaded together - or drag once per
box yourself.

Images in raw/extra/ are treated as "mixed": instead of one class for the
whole image, each box gets its own open/closed class - for photos where a
closed box and an open box both appear in frame together.

Controls:
  drag left mouse button  add a box (suggested boxes, if any, are pre-loaded
                           first - drag again to add more)
  right-click a box        (raw/extra/ only) toggle that box's class between
                           open/closed
  o / c                    (raw/extra/ only) set the class new boxes will get
                           when drawn - shown in the window title bar
  y / n / Enter            save all boxes currently drawn, move to next
  r                        undo the most recently added box
  s                        skip this image (no label saved, move on)
  b                        go back to the previous image
  q / Esc                  quit (progress already saved is kept)

"""
import argparse
import base64
import io
import json
import os
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


class BoxState:
    def __init__(self):
        self.dragging = False
        self.start = None
        self.current = None  # in-progress drag rect, not yet committed
        self.boxes = []  # committed (x1, y1, x2, y2, class_id) rects, display pixel coords


def make_mouse_callback(state, disp_w, disp_h, get_class, mixed):
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def callback(event, x, y, flags, param):
        x = clamp(x, 0, disp_w - 1)
        y = clamp(y, 0, disp_h - 1)
        if event == cv2.EVENT_LBUTTONDOWN:
            state.dragging = True
            state.start = (x, y)
            state.current = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state.dragging:
            state.current = (state.start[0], state.start[1], x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state.dragging = False
            x1, y1 = state.start
            state.current = None
            if abs(x - x1) >= 3 and abs(y - y1) >= 3:  # ignore accidental clicks
                state.boxes.append((x1, y1, x, y, get_class()))
        elif event == cv2.EVENT_RBUTTONDOWN and mixed:
            for i in range(len(state.boxes) - 1, -1, -1):  # topmost (last-drawn) box first
                bx1, by1, bx2, by2, cid = state.boxes[i]
                lo_x, hi_x = sorted((bx1, bx2))
                lo_y, hi_y = sorted((by1, by2))
                if lo_x <= x <= hi_x and lo_y <= y <= hi_y:
                    state.boxes[i] = (bx1, by1, bx2, by2, 1 - cid)
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


def full_rects_to_display(rects, orig_w, orig_h, disp_w, disp_h):
    scale_x, scale_y = disp_w / orig_w, disp_h / orig_h
    clamp_x = lambda v: max(0, min(disp_w - 1, round(v * scale_x)))
    clamp_y = lambda v: max(0, min(disp_h - 1, round(v * scale_y)))
    return [(clamp_x(x1), clamp_y(y1), clamp_x(x2), clamp_y(y2)) for x1, y1, x2, y2 in rects]


def annotate_one(base_img, disp_w, disp_h, initial_rects, source, path, class_id, window):
    mixed = class_id is None
    pending = [0]  # class new boxes get in mixed mode; toggled with o/c

    state = BoxState()
    state.boxes.extend(initial_rects)
    cv2.setMouseCallback(
        window, make_mouse_callback(state, disp_w, disp_h, lambda: class_id if not mixed else pending[0], mixed))

    while True:
        frame = base_img.copy()
        for x1, y1, x2, y2, cid in state.boxes:
            color = CLASS_COLORS[cid]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if mixed:
                cv2.putText(frame, CLASSES[cid], (x1 + 3, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        if state.current:
            x1, y1, x2, y2 = state.current
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 1)
        suggestion_note = f" [{source}: {len(initial_rects)} suggested]" if initial_rects else " [no suggestion]"
        status = f" - {len(state.boxes)} box(es), drag to add another, r=undo last"
        label = "mixed" if mixed else CLASSES[class_id]
        mode_note = f" [pending class: {CLASSES[pending[0]]} (o/c to change, right-click box to toggle)]" if mixed else ""
        cv2.putText(frame, f"{path.name} [{label}]{suggestion_note}{status}{mode_note}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window, frame)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("y"), ord("n"), 13):  # y, n, or Enter - all save+advance
            if not state.boxes:
                continue  # need at least one box before advancing
            lines = "".join(rect_to_yolo_line(b, disp_w, disp_h) for b in state.boxes)
            return ("save", lines)
        if key == ord("r"):
            if state.boxes:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="raw")
    parser.add_argument("--labels-dir", default="raw_labels")
    parser.add_argument("--provider", choices=["gemini", "openrouter"], default="openrouter",
                         help="which VLM to query for tier-1 box suggestions")
    parser.add_argument("--model", default="gemini-3.6-flash", help="Gemini model name (only used with --provider gemini)")
    parser.add_argument("--openrouter-model", default="nvidia/nemotron-nano-12b-v2-vl:free",
                         help="OpenRouter model name (only used with --provider openrouter)")
    parser.add_argument("--detector-model", default="yolo26n.pt",
                         help="generic pretrained detector used as fallback when the VLM finds nothing")
    parser.add_argument("--detector-conf", type=float, default=0.25,
                         help="confidence threshold for the fallback detector")
    parser.add_argument("--overwrite", action="store_true", help="re-annotate images that already have a label")
    parser.add_argument("--skip-vlm", action="store_true",
                         help="go straight to the local detector fallback, e.g. while the VLM's free quota is exhausted")
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
    if not pending:
        print("Nothing to do.")
        return

    window = "Auto-annotate (drag=add box, n/y/Enter=save+next, r=undo last box, s=skip, b=back, q=quit)"
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


if __name__ == "__main__":
    main()
