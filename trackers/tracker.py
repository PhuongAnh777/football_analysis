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
        
    def interpolate_ball_position(self, ball_position):
        ball_positions = [x.get(1,{}).get("bbox",[None,None,None,None]) for x in ball_position]
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.values.tolist()]

        return ball_positions

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

                # Goalkeeper counts as a field player
                if cls_name in ("player", "goalkeeper"):
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
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

            for track_id, referee in referee_dict.items():
                self.draw_ellipse(frame, referee["bbox"], (0,0,255))

            for track_id, ball in ball_dict.items():
                self.draw_traingle(frame, ball["bbox"], (0,255,0))

            self.draw_team_ball_control(frame, frame_num, team_ball_control)

        return video_frames
