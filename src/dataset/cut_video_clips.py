"""
    Usage:
        python src/cut_video_clips.py --input data_vid.mp4 \
            --segments 00:00:10-00:00:25 00:01:05-00:01:40 \
            --out-dir clips

        # merge the kept parts into a single video instead of separate clips
        python src/cut_video_clips.py --input data_vid.mp4 \
            --segments 10-25 65-100 --concat --out-dir clips
"""
import argparse
import re
from pathlib import Path

import cv2

TIME_RE = re.compile(r"^(?:(\d+):)?(?:(\d+):)?(\d+(?:\.\d+)?)$")


def parse_time(value):
    match = TIME_RE.match(value.strip())
    if not match:
        raise ValueError(f"Invalid time '{value}', expected SECONDS or [HH:]MM:SS")
    h, m, s = match.groups()
    total = float(s)
    if m is not None:
        total += float(m) * 60
    if h is not None:
        total += float(h) * 3600
    return total


def parse_segment(value):
    try:
        start_str, end_str = value.split("-")
    except ValueError:
        raise ValueError(f"Invalid segment '{value}', expected START-END, e.g. 00:00:10-00:00:25") from None
    start, end = parse_time(start_str), parse_time(end_str)
    if end <= start:
        raise ValueError(f"Segment '{value}' has end <= start")
    return start, end


def cut_segment(cap, writer, start_sec, end_sec, fps):
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    written = 0
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        written += 1
    return written


def cut_clips(input_path, segments, out_dir, concat):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Some containers (e.g. continuous camera recordings) don't carry a reliable
    # frame count, so cv2 reports a bogus value like 0 or 1. Skip the sanity
    # check rather than warn against a duration we don't actually know.
    if frame_count > 1:
        duration = frame_count / fps
        for start, end in segments:
            if start > duration or end > duration:
                print(f"Warning: segment {start:.1f}-{end:.1f}s exceeds video duration ({duration:.1f}s)")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    if concat:
        out_path = out_dir / f"{Path(input_path).stem}_cut.mp4"
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        total = 0
        for start, end in segments:
            total += cut_segment(cap, writer, start, end, fps)
        writer.release()
        print(f"Wrote {out_path} ({total} frames, ~{total / fps:.1f}s)")
    else:
        for i, (start, end) in enumerate(segments):
            out_path = out_dir / f"{Path(input_path).stem}_clip_{i:03d}.mp4"
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            written = cut_segment(cap, writer, start, end, fps)
            writer.release()
            print(f"Wrote {out_path} ({written} frames, ~{written / fps:.1f}s)")

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to source video")
    parser.add_argument("--segments", required=True, nargs="+",
                         help="segments to keep, as START-END pairs. Times can be seconds "
                              "(e.g. 12.5) or [HH:]MM:SS (e.g. 00:01:30)")
    parser.add_argument("--out-dir", default="clips")
    parser.add_argument("--concat", action="store_true",
                         help="merge all kept segments into a single output video instead "
                              "of writing one file per segment")
    args = parser.parse_args()

    parsed_segments = [parse_segment(s) for s in args.segments]
    cut_clips(args.input, parsed_segments, args.out_dir, args.concat)
