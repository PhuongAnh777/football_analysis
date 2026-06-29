from ultralytics import YOLO
import pickle
import os
import numpy as np
import cv2
import time
from utils import get_center_of_bbox, get_bbox_width, get_foot_position, blend_filled_rectangle
from .reid_extractor import DeepAppearanceExtractor
import pandas as pd

# Path to the football-tuned BoT-SORT config bundled alongside this file
_BOTSORT_CFG = os.path.join(os.path.dirname(__file__), "botsort_football.yaml")


def _bgr_color(color) -> tuple[int, int, int]:
    """OpenCV expects (B, G, R) ints; team_color may be np.float32 array."""
    if color is None:
        return (0, 0, 255)
    if isinstance(color, np.ndarray):
        vals = color.flatten()[:3]
    else:
        vals = color
    return (int(vals[0]), int(vals[1]), int(vals[2]))


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.reid  = DeepAppearanceExtractor(device="auto")

    # ── appearance ReID helpers ───────────────────────────────────────────────

    def add_appearance_to_tracks(self, tracks, frames):
        """
        Attach a 576-dim deep appearance feature to every player track entry.

        Uses MobileNetV3-Small (via ``self.reid``) instead of a colour
        histogram.  Crops in each frame are batched together for efficiency.

        The descriptor is stored as ``tracks["players"][frame_num][tid]["appearance"]``.
        """
        for frame_num, player_dict in enumerate(tracks["players"]):
            if not player_dict:
                continue
            frame  = frames[frame_num]
            tids   = list(player_dict.keys())
            bboxes = [player_dict[t]["bbox"] for t in tids]

            feats = self.reid.extract_frame(frame, bboxes)

            for tid, feat in zip(tids, feats):
                if feat is not None:
                    player_dict[tid]["appearance"] = feat

    # ── position helpers ──────────────────────────────────────────────────────

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]["position"] = position
        
    # Gaps longer than this fall back to linear interpolation (unreliable physics bridge).
    _MAX_PHYSICS_GAP_FRAMES = 45
    # Image-plane speed (px/s) above which a gap is treated as ball-in-flight.
    _PHYSICS_SPEED_THRESHOLD_PX_S = 450.0

    @staticmethod
    def _ball_bbox_row(frame_entry) -> list[float | None]:
        raw = frame_entry.get(1, {}).get("bbox") if frame_entry else None
        if raw is None or len(raw) != 4:
            return [None, None, None, None]
        return [float(v) for v in raw]

    @staticmethod
    def _bbox_from_center_halfsize(
        center: np.ndarray, half_w: float, half_h: float
    ) -> list[float]:
        cx, cy = float(center[0]), float(center[1])
        return [cx - half_w, cy - half_h, cx + half_w, cy + half_h]

    @staticmethod
    def _estimate_velocity_2d(
        centers: np.ndarray, valid: np.ndarray, idx: int, fps: float
    ) -> np.ndarray:
        """Estimate image-plane velocity (px/s) from recent valid detections."""
        if idx >= 1 and valid[idx - 1]:
            return (centers[idx] - centers[idx - 1]) * fps
        if idx >= 2 and valid[idx - 2]:
            return (centers[idx] - centers[idx - 2]) * (fps / 2.0)
        return np.zeros(2, dtype=float)

    def _fill_center_gap_physics(
        self,
        centers: np.ndarray,
        valid: np.ndarray,
        left: int,
        right: int,
        fps: float,
    ) -> None:
        """Parabolic bridge: p(t)=p0+v0*dt+0.5*a*dt^2 with endpoints at left/right."""
        p0 = centers[left].astype(float)
        p1 = centers[right].astype(float)
        dt_total = float(right - left)
        v0 = self._estimate_velocity_2d(centers, valid, left, fps)
        accel = 2.0 * (p1 - p0 - v0 * dt_total) / (dt_total ** 2)

        for t in range(left + 1, right):
            dt = float(t - left)
            centers[t] = p0 + v0 * dt + 0.5 * accel * (dt ** 2)

    def _fill_center_gap_linear(
        self, centers: np.ndarray, left: int, right: int
    ) -> None:
        p0 = centers[left].astype(float)
        p1 = centers[right].astype(float)
        for t in range(left + 1, right):
            alpha = (t - left) / float(right - left)
            centers[t] = p0 + alpha * (p1 - p0)

    @staticmethod
    def _gap_speed_px_s(
        centers: np.ndarray, valid: np.ndarray, left: int, right: int, fps: float
    ) -> float:
        """Peak image-plane speed estimate for a gap (px/s)."""
        gap = right - left
        if gap <= 0:
            return 0.0
        v_left = Tracker._estimate_velocity_2d(centers, valid, left, fps)
        speed_left = float(np.linalg.norm(v_left))
        speed_chord = float(np.linalg.norm(centers[right] - centers[left])) * fps / gap
        v_right = Tracker._estimate_velocity_2d(centers, valid, right, fps)
        speed_right = float(np.linalg.norm(v_right))
        return max(speed_left, speed_chord, speed_right)

    @staticmethod
    def _kalman_fill_centers(
        centers: np.ndarray,
        valid: np.ndarray,
        fps: float,
        *,
        measurement_noise: float = 6.0,
        process_noise: float = 12.0,
    ) -> np.ndarray:
        """Constant-velocity Kalman filter: measure at detections, predict in gaps."""
        n = len(centers)
        out = centers.copy().astype(float)
        dt = 1.0 / max(float(fps), 1.0)

        f_mat = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        h_mat = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        q_mat = np.eye(4) * process_noise
        q_mat[2, 2] = q_mat[3, 3] = process_noise * 2.5
        r_mat = np.eye(2) * measurement_noise

        state = np.zeros(4)
        cov = np.eye(4) * 500.0
        initialized = False

        for t in range(n):
            if initialized:
                state = f_mat @ state
                cov = f_mat @ cov @ f_mat.T + q_mat

            if valid[t]:
                z = centers[t].astype(float)
                if not initialized:
                    state = np.array([z[0], z[1], 0.0, 0.0])
                    cov = np.eye(4) * 100.0
                    initialized = True
                else:
                    innov = z - h_mat @ state
                    s_mat = h_mat @ cov @ h_mat.T + r_mat
                    gain = cov @ h_mat.T @ np.linalg.inv(s_mat)
                    state = state + gain @ innov
                    cov = (np.eye(4) - gain @ h_mat) @ cov
                out[t] = state[:2]
            elif initialized:
                out[t] = state[:2]

        return out

    def _interpolate_ball_centers(
        self,
        centers: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        valid: np.ndarray,
        valid_idx: np.ndarray,
        fps: float,
        *,
        method: str,
        physics_speed_threshold_px_s: float,
    ) -> None:
        """Fill missing center/size samples in-place between valid detections."""
        n = len(centers)
        first, last = int(valid_idx[0]), int(valid_idx[-1])

        for t in range(0, first):
            centers[t] = centers[first]
            half_w[t] = half_w[first]
            half_h[t] = half_h[first]
        for t in range(last + 1, n):
            centers[t] = centers[last]
            half_w[t] = half_w[last]
            half_h[t] = half_h[last]

        if method == "kalman":
            filled = self._kalman_fill_centers(centers, valid, fps)
            for t in range(n):
                if not valid[t]:
                    centers[t] = filled[t]
            return

        for i in range(len(valid_idx) - 1):
            left, right = int(valid_idx[i]), int(valid_idx[i + 1])
            if right - left <= 1:
                continue
            gap = right - left

            if gap > self._MAX_PHYSICS_GAP_FRAMES:
                use_physics = False
            elif method == "physics":
                use_physics = True
            elif method == "adaptive":
                use_physics = (
                    self._gap_speed_px_s(centers, valid, left, right, fps)
                    >= physics_speed_threshold_px_s
                )
            else:
                use_physics = False

            if use_physics:
                self._fill_center_gap_physics(centers, valid, left, right, fps)
            else:
                self._fill_center_gap_linear(centers, left, right)

            for t in range(left + 1, right):
                alpha = (t - left) / float(gap)
                half_w[t] = half_w[left] + alpha * (half_w[right] - half_w[left])
                half_h[t] = half_h[left] + alpha * (half_h[right] - half_h[left])

    def interpolate_ball_position(
        self,
        ball_position,
        fps: float = 24.0,
        *,
        method: str = "adaptive",
        physics_speed_threshold_px_s: float | None = None,
    ):
        """Fill missing ball detections.

        ``method="adaptive"`` (default): parabolic physics when estimated ball
        speed exceeds *physics_speed_threshold_px_s* (flying), else linear
        (rolling / slow).

        ``method="physics"``: always use parabolic bridge for short gaps.

        ``method="kalman"``: constant-velocity Kalman predict/update through gaps.

        ``method="linear"``: legacy pandas linear interpolation on bbox corners.
        """
        if method == "linear":
            return self._interpolate_ball_position_linear(ball_position)

        if physics_speed_threshold_px_s is None:
            physics_speed_threshold_px_s = self._PHYSICS_SPEED_THRESHOLD_PX_S

        fps = max(float(fps), 1.0)
        n = len(ball_position)
        rows = [self._ball_bbox_row(entry) for entry in ball_position]
        arr = np.array(rows, dtype=float)

        valid = ~np.isnan(arr).any(axis=1)
        if not valid.any():
            return ball_position

        centers = np.column_stack(
            ((arr[:, 0] + arr[:, 2]) / 2.0, (arr[:, 1] + arr[:, 3]) / 2.0)
        )
        half_w = (arr[:, 2] - arr[:, 0]) / 2.0
        half_h = (arr[:, 3] - arr[:, 1]) / 2.0

        valid_idx = np.flatnonzero(valid)
        self._interpolate_ball_centers(
            centers,
            half_w,
            half_h,
            valid,
            valid_idx,
            fps,
            method=method,
            physics_speed_threshold_px_s=float(physics_speed_threshold_px_s),
        )

        filled = []
        for t in range(n):
            bbox = self._bbox_from_center_halfsize(centers[t], half_w[t], half_h[t])
            filled.append({1: {"bbox": bbox}})

        return filled

    def _interpolate_ball_position_linear(self, ball_position):
        ball_positions = [
            x.get(1, {}).get("bbox", [None, None, None, None]) for x in ball_position
        ]
        df_ball_positions = pd.DataFrame(
            ball_positions, columns=["x1", "y1", "x2", "y2"]
        )
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        return [{1: {"bbox": x}} for x in df_ball_positions.values.tolist()]

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # Reset any leftover tracker state from a previous call
        self.model.predictor = None

        tracks = {"players": [], "referees": [], "ball": []}

        total = len(frames)
        print(f"[BoT-SORT] Tracking {total} frames...", flush=True)
        t0 = time.time()

        for frame_num, frame in enumerate(frames):
            # BoT-SORT: detect + track in one call.
            # persist=True keeps tracker state alive across frames.
            result = self.model.track(
                source=frame,
                tracker=_BOTSORT_CFG,
                persist=True,
                conf=0.25,
                imgsz=640,
                verbose=False,
            )[0]

            cls_names = result.names

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            if result.boxes is None or len(result.boxes) == 0:
                continue

            for box in result.boxes:
                bbox     = box.xyxy[0].tolist()
                cls_id   = int(box.cls[0])
                cls_name = cls_names.get(cls_id, "")

                # Ball has no persistent track ID; always store as ID=1
                if cls_name == "ball":
                    tracks["ball"][frame_num][1] = {"bbox": bbox}
                    continue

                if box.id is None:
                    continue
                track_id = int(box.id[0])

                # Goalkeeper counts as a field player for tracking, but role is kept.
                if cls_name in ("player", "goalkeeper"):
                    tracks["players"][frame_num][track_id] = {
                        "bbox": bbox,
                        "role": cls_name,
                    }
                elif cls_name == "referee":
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # Progress every 100 frames
            if (frame_num + 1) % 100 == 0 or frame_num == total - 1:
                elapsed = time.time() - t0
                fps_so_far = (frame_num + 1) / elapsed if elapsed > 0 else 0
                eta = (total - frame_num - 1) / fps_so_far if fps_so_far > 0 else 0
                print(
                    f"[BoT-SORT] {frame_num+1}/{total} frames"
                    f"  |  {fps_so_far:.1f} fps"
                    f"  |  ETA {eta/60:.1f} min",
                    flush=True,
                )

        if stub_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(stub_path)), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        color = _bgr_color(color)

        x_center, y_center = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center = (x_center,y2),
            axes = (int(width), int(0.35*width)),
            angle = 0.0,
            startAngle = 45,
            endAngle = 235,
            color = color,
            thickness = 2,
            lineType = cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2   
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 + rectangle_height//2) + 15
        y2_rect = y1_rect + rectangle_height

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED)
            
            x1_text = x1_rect + 12

            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_rect), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )
        return frame

    def draw_traingle(self, frame, bbox, color):
        color = _bgr_color(color)
        y = int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        traingle_points = np.array([
            [x,y], 
            [x-10,y-20], 
            [x+10,y-20]
        ])
        cv2.drawContours(frame, [traingle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [traingle_points], 0, (0,0,0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        blend_filled_rectangle(frame, (20, 620), (640, 710), alpha=0.4)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]

        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        total = team_1_num_frames + team_2_num_frames
        if total == 0:
            team_1, team_2 = 0, 0
        else:
            team_1 = team_1_num_frames / total
            team_2 = team_2_num_frames / total

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%", (30, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%", (30, 695), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        
        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        for frame_num, frame in enumerate(video_frames):
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            for track_id, player in player_dict.items():
                color = player.get("team_color", (0,0,255))
                self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get("has_ball", False):
                    self.draw_traingle(frame, player["bbox"], (0,0,255))

            for referee in referee_dict.values():
                self.draw_ellipse(frame, referee["bbox"], (0, 0, 0))

            for track_id, ball in ball_dict.items():
                self.draw_traingle(frame, ball["bbox"], (0,255,0))

            self.draw_team_ball_control(frame, frame_num, team_ball_control)

        return video_frames
