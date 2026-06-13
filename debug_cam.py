import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pickle, numpy as np
from utils.stub_io import load_track_stub

STUB = 'stubs/c428550b-a276-474c-bd3c-5cd9391784ef_track_stubs.pkl'
tracks, fps, _ = load_track_stub(STUB)
frames = tracks['players']
print(f'Frames in stub: {len(frames)},  fps: {fps}')

# Raw position_adjusted (pixel x) range
all_x, all_y = [], []
for frame in frames:
    for info in frame.values():
        pos = info.get('position_adjusted') or info.get('position')
        if pos:
            all_x.append(pos[0])
            all_y.append(pos[1])

print(f'Raw pixel x: {min(all_x):.0f} -> {max(all_x):.0f}  (span={max(all_x)-min(all_x):.0f} px)')
print(f'Raw pixel y: {min(all_y):.0f} -> {max(all_y):.0f}  (span={max(all_y)-min(all_y):.0f} px)')
print()

# Camera movement stub (old)
with open('stubs/camera_movement_stub.pkl', 'rb') as f:
    cam_old = pickle.load(f)
cam_arr = np.array(cam_old)
cum_x = np.cumsum(cam_arr[:, 0])
print(f'Old camera_movement_stub: {len(cam_old)} frames,  cum_x span = {cum_x.max()-cum_x.min():.0f} px')
print(f'  => at 0.0208 m/px: {(cum_x.max()-cum_x.min())*0.0208:.1f} m total pan')
print()

# Simulate recompute from scratch using just position data
# Average per-frame shift: how far do players shift horizontally between frames?
shifts = []
for fi in range(1, min(200, len(frames))):
    prev, curr = frames[fi-1], frames[fi]
    common = set(prev.keys()) & set(curr.keys())
    if len(common) >= 3:
        dxs = []
        for tid in common:
            p0 = (prev[tid].get('position_adjusted') or prev[tid].get('position'))
            p1 = (curr[tid].get('position_adjusted') or curr[tid].get('position'))
            if p0 and p1:
                dxs.append(p1[0] - p0[0])
        if dxs:
            shifts.append(float(np.median(dxs)))

if shifts:
    print(f'Median per-frame x-shift (first 200 frames): {np.mean(shifts):.2f} px/frame')
    print(f'  (negative = players moving left = camera panning right)')
    print(f'  Threshold in estimator: 5 px  ->  most frames BELOW threshold = pan missed!')
