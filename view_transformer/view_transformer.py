"""
ViewTransformer
===============
Maps pixel coordinates from a broadcast video to real-world pitch coordinates.

Coordinate system (top-down view, standard football-analytics convention)
-------------------------------------------------------------------------
  x-axis  (``position_transformed[0]``):
      Along pitch LENGTH — from one goal line toward the opposite goal line.
      Full FIFA-standard pitch: 0–105 m.
      Default calibration (demo video): x = 0 (left visible end) → x ≈ 23.32 m.

  y-axis  (``position_transformed[1]``):
      Across pitch WIDTH — from far touchline (opposite camera) to near
      touchline (camera side).
      Full FIFA-standard pitch: 0–68 m.

Default calibration
-------------------
The four ``pixel_vertices`` corners are calibrated for the demo video
(``input_videos/input_video.mp4``).  The visible area in that clip covers
approximately **23.32 m × 68 m** of the pitch.

To calibrate for a different video use ``PitchCalibrator``.

FIFA-standard pitch (IFAB Law 1, international matches)
    Length  : 105 m   (goal line to goal line)
    Width   :  68 m   (touchline to touchline)
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Sequence

# ── FIFA-standard pitch dimensions ──────────────────────────────────────────
PITCH_LENGTH_M = 105.0   # goal line to goal line (metres)
PITCH_WIDTH_M  =  68.0   # touchline to touchline (metres)


@dataclass
class PitchRegion:
    """Defines which portion of a 105 × 68 m pitch the camera covers.

    Parameters
    ----------
    x_min, x_max : float
        Start / end along pitch LENGTH axis (0–105 m).
        ``x_min = 0`` means the camera starts at the left goal line.
    y_min, y_max : float
        Start / end across pitch WIDTH axis (0–68 m).
        Typically 0–68 (full width visible).
    """
    x_min: float = 0.0
    x_max: float = 23.32          # visible portion in the demo calibration
    y_min: float = 0.0
    y_max: float = PITCH_WIDTH_M  # full width


class PitchCalibrator:
    """Compute a perspective transform from known pitch landmark pixels.

    Usage
    -----
    Identify the pixel positions (column, row) of at least **4** pitch
    markings whose real-world coordinates on the 105 × 68 m pitch are known.

    Common landmarks (x = along length, y = across width):

    ┌──────────────────────────────────────────────────────────┐
    │  Marking                         x (m)    y (m)          │
    │  ─────────────────────────────────────────────────────── │
    │  Left goal post – near touchline    0      68            │
    │  Left goal post – far touchline     0       0            │
    │  Left penalty area – near corner   16.5    68            │
    │  Left penalty area – far corner    16.5     0            │
    │  Halfway line – near touchline     52.5    68            │
    │  Halfway line – far touchline      52.5     0            │
    │  Center circle center             52.5    34            │
    └──────────────────────────────────────────────────────────┘

    Parameters
    ----------
    pixel_points : sequence of (col, row) pairs
        Pixel coordinates of each landmark (at least 4 required).
    world_points : sequence of (x, y) pairs
        Corresponding real-world coordinates in the 105 × 68 m system.
    """

    def __init__(
        self,
        pixel_points: Sequence[tuple[float, float]],
        world_points: Sequence[tuple[float, float]],
    ) -> None:
        if len(pixel_points) < 4 or len(world_points) < 4:
            raise ValueError("At least 4 calibration points are required.")
        if len(pixel_points) != len(world_points):
            raise ValueError(
                "pixel_points and world_points must have the same length."
            )

        px = np.array(pixel_points, dtype=np.float32)
        wx = np.array(world_points, dtype=np.float32)

        if len(pixel_points) == 4:
            self.H, _ = cv2.findHomography(px, wx)
        else:
            self.H, _ = cv2.findHomography(px, wx, cv2.RANSAC, 5.0)

        if self.H is None:
            raise RuntimeError(
                "Could not compute homography – check calibration points."
            )


class ViewTransformer:
    """Maps pixel positions to real-world pitch coordinates (metres).

    Coordinate system (see module docstring):
        pos[0] = x = along pitch LENGTH   (0 → x_max)
        pos[1] = y = across pitch WIDTH   (0 = far touchline, 68 = near/camera side)

    Parameters
    ----------
    pixel_vertices : array-like (4, 2), optional
        Four corners of the visible pitch area in the video as
        ``[[col, row], ...]``.  Defaults to the demo-video calibration.
    pitch_region : PitchRegion, optional
        Corresponding real-world region on the 105 × 68 m pitch.
        Defaults to the demo-video calibration (x: 0–23.32 m, y: 0–68 m).
    calibrator : PitchCalibrator, optional
        Pre-computed calibrator.  If supplied ``pixel_vertices`` and
        ``pitch_region`` are ignored.
    """

    # ── Default calibration — measured on 1280×720 broadcast video ────────
    # Camera: elevated midfield view (AJX vs TOT style broadcast angle).
    # Covers goal line (left) → right penalty area line (x=88.5 m).
    #
    # near-left  [51,  699] → (x=0,    y=68)  goal line,  near touchline
    # far-left   [50,  300] → (x=0,    y=0 )  goal line,  far  touchline
    # far-right  [1252, 301] → (x=88.5, y=0 )  right pen., far  touchline
    # near-right [1252, 702] → (x=88.5, y=68)  right pen., near touchline
    _DEFAULT_PIXELS_1920: np.ndarray = np.array([
        [51,   699],
        [50,   300],
        [1252, 301],
        [1252, 702],
    ], dtype=np.float32)

    # Keep _DEFAULT_PIXELS for backward compatibility
    _DEFAULT_PIXELS: np.ndarray = _DEFAULT_PIXELS_1920

    _DEFAULT_REGION: PitchRegion = PitchRegion(
        x_min=0.0, x_max=88.5, y_min=0.0, y_max=68.0
    )

    # Reference resolution the default pixels were measured at
    _REF_W: int = 1280
    _REF_H: int = 720

    def __init__(
        self,
        pixel_vertices: np.ndarray | None = None,
        pitch_region: PitchRegion | None  = None,
        calibrator: PitchCalibrator | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        frame_size : (width, height), optional
            Actual video resolution.  When provided and ``pixel_vertices`` is
            None, the default 1920×1080 calibration corners are scaled to this
            resolution automatically.  Pass ``video_frames[0].shape[1::-1]``
            (i.e. (w, h)) from the pipeline.
        """
        if calibrator is not None:
            self.perspective_transform = calibrator.H
            self.pixel_vertices        = None
        else:
            if pixel_vertices is not None:
                pv = np.array(pixel_vertices, dtype=np.float32)
            else:
                pv = self._DEFAULT_PIXELS_1920.copy()
                # Scale default corners from reference 1920×1080 to actual resolution
                if frame_size is not None:
                    fw, fh = frame_size
                    sx = fw / self._REF_W
                    sy = fh / self._REF_H
                    pv[:, 0] *= sx
                    pv[:, 1] *= sy
            pr = pitch_region or self._DEFAULT_REGION

            # Target vertices: order matches pixel_vertices
            # [near-left, far-left, far-right, near-right]
            target = np.array([
                [pr.x_min, pr.y_max],   # near touchline, left  end
                [pr.x_min, pr.y_min],   # far  touchline, left  end
                [pr.x_max, pr.y_min],   # far  touchline, right end
                [pr.x_max, pr.y_max],   # near touchline, right end
            ], dtype=np.float32)

            self.perspective_transform = cv2.getPerspectiveTransform(pv, target)
            self.pixel_vertices        = pv

    def pan_scale_mpp(self) -> float:
        """Metres per pixel for a horizontal camera pan.

        Estimated by transforming two horizontally adjacent pixels at the
        vertical centre of the calibration region and measuring how many
        world-x metres separate them.  Returns a positive scalar (m / px).

        For the default demo calibration the value is roughly 0.020 m/px
        (matches the geometric mean of near/far touchline scales).
        """
        if self.pixel_vertices is None:
            return 0.02   # fallback for calibrator-based init

        # Vertical centre of calibration region
        ys   = self.pixel_vertices[:, 1]
        cy   = float(np.mean(ys))
        # Horizontal centre
        xs   = self.pixel_vertices[:, 0]
        cx   = float(np.mean(xs))

        p1 = self.transform_point(np.array([cx,     cy], dtype=np.float32))
        p2 = self.transform_point(np.array([cx + 1, cy], dtype=np.float32))
        if p1 is None or p2 is None:
            # Fallback: use near-touchline calibration ratio
            near_left  = self.pixel_vertices[0]   # [110, 1035]
            near_right = self.pixel_vertices[3]   # [1640, 915]
            pr = self._DEFAULT_REGION
            px_span = float(abs(near_right[0] - near_left[0]))
            m_span  = float(abs(pr.x_max - pr.x_min))
            return m_span / max(px_span, 1.0)

        return float(abs(p2[0, 0] - p1[0, 0]))

    def compute_pitch_offsets(
        self,
        cumulative_camera_movement: list,
        pitch_x_start: float = 0.0,
    ) -> list:
        """Convert cumulative camera-movement deltas to per-frame pitch offsets.

        Parameters
        ----------
        cumulative_camera_movement : list of [cum_x, cum_y]
            Output of ``CameraMovementEstimator.cumulative()``.
            cum_x < 0 means camera panned RIGHT (toward higher pitch x).
        pitch_x_start : float
            World x (metres) that the LEFT edge of the camera corresponds to
            at frame 0.  0.0 = left goal line.  52.5 = halfway line, etc.

        Returns
        -------
        list of float
            pitch_offset_x[frame] = where the local x=0 maps to on the
            105 m pitch.  Add this to position_transformed[0] (local x).
        """
        scale = self.pan_scale_mpp()
        offsets = []
        for cum_x, _cum_y in cumulative_camera_movement:
            # Camera panned right → cum_x < 0 → offset increases (camera
            # is now looking at a higher-x portion of the pitch).
            offset = pitch_x_start + (-cum_x) * scale
            offsets.append(offset)
        return offsets

    def transform_point(self, point: np.ndarray) -> np.ndarray | None:
        """Transform a single pixel (col, row) to world (x, y) coordinates.

        Returns ``None`` if the point lies outside the calibrated region.
        """
        if self.pixel_vertices is not None:
            p = (int(point[0]), int(point[1]))
            if cv2.pointPolygonTest(self.pixel_vertices, p, False) < 0:
                return None
        reshaped = point.reshape(-1, 1, 2).astype(np.float32)
        result   = cv2.perspectiveTransform(reshaped, self.perspective_transform)
        return result.reshape(-1, 2)

    def add_transformed_position_to_tracks(
        self,
        tracks: dict,
        pitch_offsets: list | None = None,
    ) -> None:
        """Add ``"position_transformed"`` to every track entry in-place.

        Parameters
        ----------
        tracks : dict
            Full pipeline tracks dict.
        pitch_offsets : list of float, optional
            Per-frame pitch x offset (metres) computed by
            ``compute_pitch_offsets()``.  When supplied, the world-x
            coordinate becomes ``local_x + pitch_offsets[frame]``, placing
            players on the full 105 m pitch rather than the local 0–23.32 m
            window.  When ``None`` (default) behaviour is unchanged.
        """
        for object_tracks in tracks.values():
            for frame_num, frame in enumerate(object_tracks):
                offset_x = (
                    float(pitch_offsets[frame_num])
                    if pitch_offsets is not None
                    and frame_num < len(pitch_offsets)
                    else 0.0
                )
                for track_info in frame.values():
                    pos = track_info.get("position_adjusted")
                    if pos is None:
                        track_info["position_transformed"] = None
                        continue
                    transformed = self.transform_point(np.array(pos))
                    if transformed is None:
                        track_info["position_transformed"] = None
                    else:
                        local = transformed.squeeze().tolist()
                        # local[0] = x (pitch depth), local[1] = y (pitch width)
                        track_info["position_transformed"] = [
                            local[0] + offset_x,
                            local[1],
                        ]
