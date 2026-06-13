import pickle
import cv2
import numpy as np
import os
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator:
    def __init__(self, frame):
        self.minimum_distance = 2
        self.lk_params = dict (
            winSize = (15, 15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20]    = 1
        mask_features[:, 900:1500] = 1

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 7,
            blockSize = 7,
            mask = mask_features,
        )
    
    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info["position"]
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0] + camera_movement[0], position[1] + camera_movement[1])
                    tracks[object][frame_num][track_id]["position_adjusted"] = position_adjusted

    def get_camera_movement(self, frames, read_from_stub = False, stub_path = None):
        # Read from stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
        
        camera_movement = [[0, 0] for _ in range(len(frames))]

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        # Shi-Tomasi Corner Detector.
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        # Lucas-Kanade Optical Flow
        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _,_ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            # Collect all valid feature displacements
            dxs, dys, dists = [], [], []
            for new_pt, old_pt in zip(new_features, old_features):
                np_ = new_pt.ravel()
                op_ = old_pt.ravel()
                dist = measure_distance(np_, op_)
                dx, dy = measure_xy_distance(np_, op_)
                dxs.append(dx); dys.append(dy); dists.append(dist)

            if dists:
                max_distance = float(np.max(dists))
            else:
                max_distance = 0

            if max_distance > self.minimum_distance:
                # Use median to ignore fast-moving players / outliers
                camera_movement_x = float(np.median(dxs))
                camera_movement_y = float(np.median(dys))
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            
            old_gray = frame_gray.copy()

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)
        
        return camera_movement

    @staticmethod
    def cumulative(camera_movement_per_frame):
        """Convert per-frame camera deltas → cumulative offsets from frame 0.

        camera_movement_per_frame : list of [dx, dy]
            Per-frame feature displacement (new_feature - old_feature).
            Negative dx = camera panned right (features moved left).

        Returns
        -------
        list of [cum_x, cum_y] – same length as input.
            cum_x < 0 means camera has panned right (toward higher-x pitch).
        """
        result = [[0.0, 0.0]]
        cum_x, cum_y = 0.0, 0.0
        for dx, dy in camera_movement_per_frame[1:]:
            cum_x += dx
            cum_y += dy
            result.append([cum_x, cum_y])
        return result

    # draw_camera_movement removed — camera offset vẫn dùng nội bộ qua add_adjust_positions_to_tracks
