#!/usr/bin/env python3
import argparse, os, re, signal, sys, time
import threading
import subprocess
import numpy as np
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import cv2
from ultralytics import YOLO
import queue

def is_net(src: str) -> bool:
    return bool(re.match(r"^(rtmp|rtsp|http)s?://", str(src), re.I))







class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        from filterpy.kalman import KalmanFilter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def _convert_bbox_to_z(self, bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h) if h != 0 else 1.0
        return np.array([x, y, s, r]).reshape((4, 1))

    def _convert_x_to_bbox(self, x):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w if w != 0 else 0
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))

    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return self._convert_x_to_bbox(self.kf.x)


def iou_batch(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o


class Sort:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = self._associate(dets, trks)
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])
            self.trackers.append(trk)
        ret = []
        for trk in self.trackers:
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                d = trk.get_state()[0]
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
        self.trackers = [t for t in self.trackers if t.time_since_update < self.max_age]
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def _associate(self, detections, trackers):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        iou_matrix = iou_batch(detections[:, :4], trackers[:, :4])
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(row_ind, col_ind)))
        if len(matched_indices) > 0:
            unmatched_dets = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
            unmatched_trks = [t for t in range(len(trackers)) if t not in matched_indices[:, 1]]
        else:
            unmatched_dets = []
            unmatched_trks = []
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_dets.append(m[0])
                unmatched_trks.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        return matches, np.array(unmatched_dets), np.array(unmatched_trks)


def parse_args():
    p = argparse.ArgumentParser("YOLO + SORT + RTMP Output")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--source", type=str, default='rtmp://nginx-rtmp:1935/live/stream')
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.50)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--max-det", type=int, default=50)
    p.add_argument("--classes", type=str, default="")
    p.add_argument("--max-age", type=int, default=30)
    p.add_argument("--min-hits", type=int, default=3)
    p.add_argument("--iou-threshold", type=float, default=0.3)
    p.add_argument("--max-area-frac", type=float, default=0.25)
    p.add_argument("--output-rtmp", type=str, required=True, help="RTMP output URL")  # CHANGED
    p.add_argument("--fps", type=int, default=25, help="Output FPS")
    return p.parse_args()


class RTMPStreamer:
    def __init__(self, rtmp_url, width, height, fps=30):
        self.rtmp_url = rtmp_url
        self.width = width
        self.height = height
        self.fps = fps
        self.process = None
        self.frame_queue = queue.Queue(maxsize=10)  # Limit queue size
        self.running = False

    def start(self):
        cmd = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', '-',  # Read from stdin
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-profile:v', 'baseline',
            '-pix_fmt', 'yuv420p',
            '-b:v', '3000k',  # Bitrate limit
            '-maxrate', '3000k',
            '-bufsize', '6000k',
            '-g', str(self.fps * 2),  # Keyframe interval
            '-f', 'flv',
            self.rtmp_url
        ]

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10 ** 8  # Large buffer
        )

        self.running = True

        # Start thread to consume stderr (prevent blocking)
        self.stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self.stderr_thread.start()

        # Start thread to write frames
        self.write_thread = threading.Thread(target=self._write_frames, daemon=True)
        self.write_thread.start()

        print(f"[RTMP] Streaming to {self.rtmp_url}")

    def _read_stderr(self):
        """Consume stderr to prevent pipe blocking"""
        while self.running:
            try:
                line = self.process.stderr.readline()
                if not line:
                    break
                # Optionally print errors
                # print(f"[FFmpeg] {line.decode().strip()}")
            except:
                break

    def _write_frames(self):
        """Write frames from queue to FFmpeg stdin"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                if frame is None:  # Poison pill
                    break
                self.process.stdin.write(frame.tobytes())
            except queue.Empty:
                continue
            except BrokenPipeError:
                print("[RTMP] Broken pipe - FFmpeg died")
                break
            except Exception as e:
                print(f"[RTMP] Write error: {e}")
                break

    def write_frame(self, frame):
        """Add frame to queue (non-blocking)"""
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            # Drop frame if queue is full (prevents blocking)
            pass

    def stop(self):
        """Stop streaming"""
        self.running = False

        # Send poison pill
        try:
            self.frame_queue.put(None, timeout=1)
        except:
            pass

        # Close stdin
        if self.process and self.process.stdin:
            try:
                self.process.stdin.close()
            except:
                pass

        # Wait for process
        if self.process:
            try:
                self.process.wait(timeout=2)
            except:
                self.process.kill()

        print("[RTMP] Streaming stopped")
class HLSWriter:
    """Write frames to HLS using FFmpeg"""

    def __init__(self, output_path, width, height, fps=25):
        self.output_path = output_path
        self.process = None

        # FFmpeg command for HLS output
        cmd = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-f', 'hls',
            '-hls_time', '2',
            '-hls_list_size', '5',
            '-hls_flags', 'delete_segments',
            output_path
        ]

        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        print(f"[HLS] Started streaming to {output_path}")

    def write(self, frame):
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(frame.tobytes())
            except:
                pass

    def release(self):
        if self.process:
            self.process.stdin.close()
            self.process.wait()


def create_waiting_frame(width=1280, height=720, message="Waiting for input..."):
    """Create a placeholder frame"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(message, font, 1.0, 2)
    x = (width - tw) // 2
    y = (height + th) // 2
    cv2.putText(frame, message, (x, y), font, 1.0, (0, 165, 255), 2)
    return frame


def main():
    args = parse_args()

    # Load model
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model not found: {args.weights}")
    if weights_path.is_file():
        weights_path = weights_path.parent

    print(f"[INFO] Loading model: {weights_path}")
    model = YOLO(str(weights_path), task='detect')

    # SORT tracker
    tracker = Sort(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_threshold)

    # Parse classes
    classes = None
    if args.classes.strip():
        classes = [int(x) for x in args.classes.split(",") if x.strip().isdigit()]

    # Color palette
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(200, 3), dtype=np.uint8)

    # Initialize RTMP writer FIRST with default dimensions
    default_width = 1280
    default_height = 720

    print(f"[INFO] Starting RTMP output to: {args.output_rtmp}")
    rtmp_writer = RTMPStreamer(args.output_rtmp, default_width, default_height, args.fps)
    rtmp_writer.start()

    # Create waiting frame
    waiting_frame = create_waiting_frame(default_width, default_height)

    cap = None
    frame_count = 0
    stream_active = False

    print(f"[INFO] Connecting to input: {args.source}")

    try:
        while True:
            # Try to connect if not connected
            if cap is None or not cap.isOpened():
                if stream_active:
                    print("[WARN] Input stream lost, switching to waiting mode...")
                    stream_active = False

                cap = cv2.VideoCapture(args.source, cv2.CAP_FFMPEG)

                if not cap.isOpened():
                    # Send waiting frame
                    rtmp_writer.write_frame(waiting_frame)
                    time.sleep(0.04)  # ~25fps
                    continue

                # Get stream properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or args.fps

                print(f"[INFO] Input connected: {width}x{height} @ {fps:.1f} FPS")

                # Restart writer with correct dimensions if needed
                if width != rtmp_writer.width or height != rtmp_writer.height:
                    print(f"[INFO] Restarting RTMP writer with new dimensions...")
                    rtmp_writer.stop()
                    rtmp_writer = RTMPStreamer(args.output_rtmp, width, height, args.fps)
                    rtmp_writer.start()
                    waiting_frame = create_waiting_frame(width, height)

                stream_active = True
                frame_count = 0

            # Try to read frame
            ret, frame = cap.read()
            if not ret:
                # Send waiting frame on failure
                rtmp_writer.write_frame(waiting_frame)

                if stream_active:
                    print("[WARN] Failed to read frame, reconnecting...")
                    stream_active = False

                cap.release()
                cap = None
                time.sleep(1)
                continue

            frame_count += 1

            # Run detection
            results = model(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                classes=classes,
                verbose=False,
            )

            # Extract detections
            boxes = results[0].boxes
            if len(boxes) > 0:
                dets = boxes.xyxy.cpu().numpy()
                scores = boxes.conf.cpu().numpy()

                # Filter by area
                if args.max_area_frac > 0:
                    frame_area = frame.shape[0] * frame.shape[1]
                    areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
                    valid_mask = areas <= (frame_area * args.max_area_frac)
                    dets = dets[valid_mask]
                    scores = scores[valid_mask]

                if len(dets) > 0:
                    dets_with_score = np.column_stack((dets, scores))
                    tracks = tracker.update(dets_with_score)
                else:
                    tracks = tracker.update(np.empty((0, 5)))
            else:
                tracks = tracker.update(np.empty((0, 5)))

            # Draw tracks
            output_frame = frame.copy()
            for track in tracker.trackers:
                d = track.get_state()[0]
                track_id = int(track.id)
                x1, y1, x2, y2 = map(int, d)

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                color = tuple(map(int, colors[track_id % len(colors)]))
                thickness = 2 if track.hit_streak >= args.min_hits else 1

                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, thickness)

                label = f"ID:{track_id}"
                if track.time_since_update > 0:
                    label += f" ({track.time_since_update})"

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(output_frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
                cv2.putText(output_frame, label, (x1 + 2, y1 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Add info overlay
            info = f"Frame: {frame_count} | Tracks: {len(tracker.trackers)}"
            cv2.putText(output_frame, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Write to RTMP (always sends frame)
            rtmp_writer.write_frame(output_frame)

            if frame_count % 100 == 0:
                print(f"[INFO] Processed {frame_count} frames, {len(tracker.trackers)} active tracks")

    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    finally:
        if cap is not None:
            cap.release()
        rtmp_writer.stop()
        print("[INFO] Stopped")


if __name__ == "__main__":
    main()