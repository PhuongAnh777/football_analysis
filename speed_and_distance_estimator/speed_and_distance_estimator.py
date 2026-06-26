import cv2
from utils import measure_distance, get_foot_position

class SpeedAndDistance_Estimator:
    def __init__(self, fps: int = 24, frame_window: int = 5):
        self.frame_window = frame_window
        self.frame_rate = max(1, int(fps))
    
    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}

        for object, object_tracks in tracks.items():
            if object == 'ball' or object == 'referees':
                continue

            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min (frame_num + self.frame_window, number_of_frames - 1)

                for track_id,_ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id]["position_transformed"]
                    end_position = object_tracks[last_frame][track_id]["position_transformed"]

                    if start_position is None or end_position is None:
                        continue

                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    speed_meteres_per_second = distance_covered / time_elapsed
                    speed_km_per_hour = speed_meteres_per_second * 3.6

                    if object not in total_distance:
                        total_distance[object] = {}

                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0

                    total_distance[object][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue

                        tracks[object][frame_num_batch][track_id]["speed"] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]["distance"] = total_distance[object][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == 'ball' or object == 'referees':
                    continue
                for _, track_info in object_tracks[frame_num].items():
                    if "speed" not in track_info:
                        continue

                    speed = track_info.get("speed")
                    distance = track_info.get("distance")
                    if speed is None and distance is None:
                        continue

                    bbox = track_info.get("bbox")
                    position = list(get_foot_position(bbox))
                    position[1] += 40
                    position = tuple(map(int, position))

                    font_scale = 0.45
                    thickness = 1
                    line_gap = 12
                    x, y = position[0] + 4, position[1]
                    for text, dy in (
                        (f"{speed:.0f} km/h", 0),
                        (f"{distance:.0f} m", line_gap),
                    ):
                        cv2.putText(
                            frame, text, (x, y + dy),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (0, 0, 0), thickness + 1, cv2.LINE_AA,
                        )
                        cv2.putText(
                            frame, text, (x, y + dy),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (255, 255, 255), thickness, cv2.LINE_AA,
                        )

        return frames
