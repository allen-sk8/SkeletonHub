#!/usr/bin/env python
"""
SkeletonHub 2D Human Bounding Box Preprocessor

Runs bounding box extraction and tracking over a video using the chosen tracking algorithm
(yolov8, yolov11, rtmdet, skater_short, or skater_long), then saves coordinates to a text file
and optionally generates a visualization overlay video.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import subprocess
from pathlib import Path

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from preprocessors.trackers.standard_tracker import StandardTracker
from preprocessors.trackers.skater_short import SkaterShortTracker
from preprocessors.trackers.skater_long import SkaterLongTracker

def main():
    parser = argparse.ArgumentParser(description="SkeletonHub 2D Human BBox Preprocessor")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", help="Path to output .txt file. If omitted, saves to data/bbox_cache/<video_stem>_bbox.txt")
    parser.add_argument("--det-type", default="yolov8", choices=["yolov8", "yolov11", "rtmdet", "skater_short", "skater_long"],
                        help="Bounding box detector/tracker algorithm type")
    parser.add_argument("--weights", help="Optional custom weights file for YOLO models")
    parser.add_argument("--tracker-cfg", help="Optional custom tracker configuration yaml (for skater_short/skater_long)")
    parser.add_argument("--device", default="cuda:0", help="GPU or CPU device index (default: cuda:0)")
    parser.add_argument("--no-vis", action="store_true", help="Skip generating visualization video overlay")
    parser.add_argument("--vis-output", help="Path to output visualization video file")
    args = parser.parse_args()

    # 1. Resolve paths
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input video not found: {args.input}")
        sys.exit(1)

    output_path = args.output
    if not output_path:
        cache_dir = Path(project_root) / "data" / "bbox_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_path = cache_dir / f"{input_path.stem}_bbox.txt"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Select Tracker
    print(f"🚀 Initializing tracking algorithm: {args.det_type} on device {args.device}...")
    if args.det_type in ["yolov8", "yolov11", "rtmdet"]:
        tracker = StandardTracker(det_type=args.det_type, weights=args.weights, device=args.device)
    elif args.det_type == "skater_short":
        tracker = SkaterShortTracker(weights=args.weights, tracker_cfg=args.tracker_cfg, device=args.device)
    elif args.det_type == "skater_long":
        tracker = SkaterLongTracker(weights=args.weights, device=args.device)
    else:
        raise ValueError(f"❌ Unknown detector type: {args.det_type}")

    # 3. Execute tracking
    print(f"🎥 Running tracking on video: {input_path}...")
    try:
        bboxes = tracker.track(str(input_path))
    except Exception as e:
        print(f"❌ Error during tracking: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 4. Save coordinates
    print(f"💾 Saving bounding box coordinates to: {output_path}...")
    with open(output_path, 'w') as f:
        for box in bboxes:
            f.write(f"{box[0]:.1f} {box[1]:.1f} {box[2]:.1f} {box[3]:.1f}\n")
    print(f"✅ Saved BBoxes of shape: {bboxes.shape}")

    # 5. Optional Visualization Overlay
    if not args.no_vis:
        vis_out = args.vis_output
        if not vis_out:
            vis_out = output_path.parent / f"{output_path.stem}_vis.mp4"
        vis_out = Path(vis_out)
        vis_out.parent.mkdir(parents=True, exist_ok=True)

        print(f"🎬 Creating visualization overlay: {vis_out}...")
        cap = cv2.VideoCapture(str(input_path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        temp_vis = str(vis_out) + ".raw.mp4"
        out = cv2.VideoWriter(temp_vis, fourcc, fps, (w, h))

        frame_idx = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_idx < len(bboxes):
                box = bboxes[frame_idx]
                # If box is valid (not all -1)
                if not (box[0] == -1.0 and box[1] == -1.0):
                    x1, y1, x2, y2 = map(int, box)
                    # Draw purple rectangle for final tracking box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 0, 128), 3)
                    # Put frame number and class label
                    cv2.putText(frame, f"F:{frame_idx} | Person", (x1, max(y1 - 10, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 2)
            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        # FFMPEG re-encoding to libx264/yuv420p for standard player compatibility
        print(f"🔄 Re-encoding visualization video via ffmpeg...")
        subprocess.run(["ffmpeg", "-y", "-i", temp_vis,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                        "-c:a", "copy", str(vis_out)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if os.path.exists(temp_vis):
            os.remove(temp_vis)
        print(f"✅ Visualization video saved to: {vis_out}")

    print("🎉 BBox preprocessing completed successfully.")

if __name__ == "__main__":
    main()
