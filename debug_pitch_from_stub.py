"""
Tính kích thước sân từ stub có sẵn mà không cần chạy lại pipeline.
Dùng camera_movement_stub.pkl + tracking stub mới nhất.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os, pickle
import numpy as np
from utils.stub_io import load_track_stub
from view_transformer import ViewTransformer
from camera_movement_estimator import CameraMovementEstimator

TRACK_STUB = 'stubs/c428550b-a276-474c-bd3c-5cd9391784ef_track_stubs.pkl'
CAM_STUB   = 'stubs/camera_movement_stub.pkl'

# ── 1. Load tracks ───────────────────────────────────────────────────────────
tracks, fps, _ = load_track_stub(TRACK_STUB)
n_frames = len(tracks['players'])
print(f'Track stub  : {n_frames} frames @ {fps} fps')

# ── 2. Load & truncate camera movement ──────────────────────────────────────
with open(CAM_STUB, 'rb') as f:
    cam_raw = pickle.load(f)

cam_raw_trunc = cam_raw[:n_frames]
while len(cam_raw_trunc) < n_frames:
    cam_raw_trunc.append([0, 0])

print(f'Camera stub : {len(cam_raw)} frames  (truncated to {n_frames})')

# ── 3. Cumulative movement ───────────────────────────────────────────────────
cum = CameraMovementEstimator.cumulative(cam_raw_trunc)
cum_arr = np.array(cum)
cum_x_span = float(cum_arr[:, 0].max() - cum_arr[:, 0].min())

print(f'cum_x span  : {cum_x_span:.0f} px')

# ── 4. ViewTransformer + offsets ─────────────────────────────────────────────
vt = ViewTransformer()
mpp = vt.pan_scale_mpp()
offsets = vt.compute_pitch_offsets(cum, pitch_x_start=0.0)
offset_span = max(offsets) - min(offsets)

print(f'm_per_px    : {mpp:.4f}')
print(f'offset span : {offset_span:.1f} m  (camera lia duoc)')

# ── 5. Apply transform ──────────────────────────────────────────────────────
# Reset existing transforms first
for frame in tracks['players']:
    for info in frame.values():
        info.pop('position_transformed', None)

# Rebuild position_adjusted from raw position (not adjusted)
for frame in tracks['players']:
    for info in frame.values():
        pos = info.get('position')
        if pos:
            info['position_adjusted'] = pos

vt.add_transformed_position_to_tracks(tracks, pitch_offsets=offsets)

all_x, all_y = [], []
for frame in tracks['players']:
    for info in frame.values():
        pt = info.get('position_transformed')
        if pt is not None:
            all_x.append(pt[0])
            all_y.append(pt[1])

if all_x:
    x_span = max(all_x) - min(all_x)
    y_span = max(all_y) - min(all_y)
    print()
    print(f'[Pitch] Kich thuoc san do duoc:')
    print(f'  dai  = {x_span:.1f} m   ({min(all_x):.1f} -> {max(all_x):.1f} m)')
    print(f'  rong = {y_span:.1f} m   ({min(all_y):.1f} -> {max(all_y):.1f} m)')
    print()
    if x_span >= 80:
        print('[OK]  Chieu dai hop ly (>= 80 m)')
    elif x_span >= 50:
        print('[WARN] Chieu dai trung binh (50-80 m) — camera co the khong lia het san')
    else:
        print('[FAIL] Chieu dai qua nho (<50 m) — calibration hoac camera movement sai')
