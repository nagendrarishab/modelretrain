"""
Requires a Gemini API key in a .env file at the project root

Pass --skip-gemini to go straight to the generic detector fallback - no key
needed, and skips the network round-trip + retries entirely. Useful while
Gemini's daily free-tier quota is exhausted (manual work)

Controls:
  y                       accept the suggested box as-is, move to next
  drag left mouse button  redraw the box (overrides the suggestion)
  n / Enter               save the current box and move to the next image
  r                       clear the box on this image and redraw
  s                       skip this image (no label saved, move on)
  b                       go back to the previous image
  q / Esc                 quit (progress already saved is kept)

If neither Gemini nor the generic detector finds a box, the review window
opens empty so you can draw it manually.

Already-labeled images are skipped on the next run unless --overwrite is
passed, so annotation can be resumed across sessions.
"""
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageOps
from ultralytics import YOLO

CLASSES = ["closed", "open"] 
MAX_DISPLAY = 900


class BoxState:
    def __init__(self):
        self.dragging = False
        self.start = None
        self.rect = None 

    def reset(self):
        self.dragging = False
        self.start = None
        self.rect = None


def make_mouse_callback(state, disp_w, disp_h):
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def callback(event, x, y, flags, param):
        x = clamp(x, 0, disp_w - 1)
        y = clamp(y, 0, disp_h - 1)
        if event == cv2.EVENT_LBUTTONDOWN:
            state.dragging = True
            state.start = (x, y)
            state.rect = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state.dragging:
            state.rect = (state.start[0], state.start[1], x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state.dragging = False
            state.rect = (state.start[0], state.start[1], x, y)

    return callback


def rect_to_yolo_line(rect, disp_w, disp_h, class_id):
    x1, y1, x2, y2 = rect
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    cx = (x1 + x2) / 2 / disp_w
    cy = (y1 + y2) / 2 / disp_h
    w = (x2 - x1) / disp_w
    h = (y2 - y1) / disp_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"


PROMPT = """Find the plastic storage box/container (it has a flat lid and may be \
open or closed) in this image. Respond with ONLY a JSON array, no other text.

If the box is visible, the array has exactly one object:
[{"box_2d": [ymin, xmin, ymax, xmax]}]
where each coordinate is normalized to 0-1000 relative to image height/width.

If no such box is visible anywhere in the image, respond with an empty array: []
"""


def load_full_image(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    return img, img.width, img.height


def query_gemini_box(client, model, image, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[image, PROMPT],
                config=types.GenerateContentConfig(temperature=0),
            )
            text = response.text.strip()
            if text.startswith("```"):
                text = text.strip("`")
                text = text[text.find("[") :]
            boxes = json.loads(text)
            if not boxes:
                return None
            ymin, xmin, ymax, xmax = boxes[0]["box_2d"]
            x1 = xmin / 1000 * image.width
            y1 = ymin / 1000 * image.height
            x2 = xmax / 1000 * image.width
            y2 = ymax / 1000 * image.height
            return (x1, y1, x2, y2)
        except Exception as e:
            wait = 2**attempt
            print(f"    Gemini request failed ({e}); retrying in {wait}s..." if attempt + 1 < max_retries
                  else f"    Gemini request failed ({e}); giving up, leave box empty.")
            if attempt + 1 < max_retries:
                time.sleep(wait)
    return None


def query_generic_detector(detector, image, conf):
    results = detector.predict(image, conf=conf, verbose=False)[0]
    if len(results.boxes) == 0:
        return None
    best = int(results.boxes.conf.argmax())
    x1, y1, x2, y2 = results.boxes.xyxy[best].tolist()
    return (x1, y1, x2, y2)


def full_rect_to_display(rect, orig_w, orig_h, disp_w, disp_h):
    if rect is None:
        return None
    scale_x, scale_y = disp_w / orig_w, disp_h / orig_h
    x1, y1, x2, y2 = rect
    clamp_x = lambda v: max(0, min(disp_w - 1, round(v * scale_x)))
    clamp_y = lambda v: max(0, min(disp_h - 1, round(v * scale_y)))
    return (clamp_x(x1), clamp_y(y1), clamp_x(x2), clamp_y(y2))


def annotate_one(base_img, disp_w, disp_h, initial_rect, source, path, class_id, window):
    state = BoxState()
    state.rect = initial_rect
    cv2.setMouseCallback(window, make_mouse_callback(state, disp_w, disp_h))

    while True:
        frame = base_img.copy()
        if state.rect:
            x1, y1, x2, y2 = state.rect
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        suggestion_note = f" [{source} suggestion - y=accept]" if initial_rect else " [no suggestion - draw manually]"
        cv2.putText(frame, f"{path.name} [{CLASSES[class_id]}]{suggestion_note}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window, frame)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("y") and initial_rect:
            return ("save", rect_to_yolo_line(initial_rect, disp_w, disp_h, class_id))
        if key in (ord("n"), 13):  # n or Enter
            if state.rect is None:
                continue  # need a box before advancing
            return ("save", rect_to_yolo_line(state.rect, disp_w, disp_h, class_id))
        if key == ord("r"):
            state.reset()
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
    parser.add_argument("--model", default="gemini-3.6-flash", help="Gemini model to query for box suggestions")
    parser.add_argument("--detector-model", default="yolo26n.pt",
                         help="generic pretrained detector used as fallback when Gemini finds nothing")
    parser.add_argument("--detector-conf", type=float, default=0.25,
                         help="confidence threshold for the fallback detector")
    parser.add_argument("--overwrite", action="store_true", help="re-annotate images that already have a label")
    parser.add_argument("--skip-gemini", action="store_true",
                         help="go straight to the local detector fallback, e.g. while Gemini's daily quota is exhausted")
    args = parser.parse_args()

    client = None
    if not args.skip_gemini:
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("Set GEMINI_API_KEY in a .env file, "
                              "or pass --skip-gemini to use only the local detector.")
        client = genai.Client(api_key=api_key)
    detector = YOLO(args.detector_model)

    raw_dir = Path(args.raw_dir)
    labels_dir = Path(args.labels_dir)

    items = []
    for class_id, cls in enumerate(CLASSES):
        files = sorted((raw_dir / cls).glob("*.jpg")) + sorted((raw_dir / cls).glob("*.jpeg"))
        for f in files:
            items.append((f, class_id, cls))

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

    window = "Auto-annotate (y=accept suggestion, drag=redraw, n=save+next, r=redo, s=skip, b=back, q=quit)"
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

        full_rect, source = None, None
        if not args.skip_gemini:
            print("    querying Gemini...")
            full_rect = query_gemini_box(client, args.model, full_img)
            source = "Gemini"
            if full_rect is None:
                print("    Gemini found nothing; trying generic detector...")
        if full_rect is None:
            full_rect = query_generic_detector(detector, full_img, args.detector_conf)
            source = "generic detector"
        if full_rect is None:
            print("    generic detector found nothing either; draw manually.")
            source = None
        initial_rect = full_rect_to_display(full_rect, orig_w, orig_h, disp_w, disp_h)

        action, payload = annotate_one(base_img, disp_w, disp_h, initial_rect, source, path, class_id, window)

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
