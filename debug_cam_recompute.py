"""
Test camera movement với code mới (threshold=2 + median) trên video thực.
Chay: python debug_cam_recompute.py [video_path]
Mac dinh: input_videos/input_video.mp4
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import cv2, numpy as np, time

VIDEO = sys.argv[1] if len(sys.argv) > 1 else 'input_videos/input_video.mp4'

print(f'Video: {VIDEO}')

# ── Doc video ──────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO)
fps   = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'{total} frames @ {fps:.0f} fps  ({total/fps:.1f}s)')

frames = []
print('Doc video...', end='', flush=True)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
    if len(frames) % 500 == 0:
        print(f' {len(frames)}', end='', flush=True)
cap.release()
print(f' xong ({len(frames)} frames)')

# ── Camera Movement Estimator (code moi) ──────────────────────────────────
from utils import measure_distance, measure_xy_distance

MINIMUM_DISTANCE = 2   # da ha tu 5 xuong 2

first_frame_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(first_frame_gray)
mask[:, 0:20]      = 1
mask[:, 900:1500]  = 1

features_params = dict(maxCorners=100, qualityLevel=0.3,
                       minDistance=7, blockSize=7, mask=mask)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

camera_movement = [[0.0, 0.0] for _ in range(len(frames))]
old_gray     = first_frame_gray.copy()
old_features = cv2.goodFeaturesToTrack(old_gray, **features_params)

n_detected = 0
t0 = time.time()
print('Tinh camera movement...', end='', flush=True)

for fi in range(1, len(frames)):
    frame_gray   = cv2.cvtColor(frames[fi], cv2.COLOR_BGR2GRAY)
    new_features, _, _ = cv2.calcOpticalFlowPyrLK(
        old_gray, frame_gray, old_features, None, **lk_params)

    dxs, dys, dists = [], [], []
    for npt, opt in zip(new_features, old_features):
        n_ = npt.ravel(); o_ = opt.ravel()
        d  = measure_distance(n_, o_)
        dx, dy = measure_xy_distance(n_, o_)
        dxs.append(dx); dys.append(dy); dists.append(d)

    max_dist = float(np.max(dists)) if dists else 0.0

    if max_dist > MINIMUM_DISTANCE:
        camera_movement[fi] = [float(np.median(dxs)), float(np.median(dys))]
        old_features = cv2.goodFeaturesToTrack(frame_gray, **features_params)
        n_detected += 1

    old_gray = frame_gray.copy()

    if fi % 500 == 0:
        print(f' {fi}', end='', flush=True)

print(f' xong  ({time.time()-t0:.0f}s)')
print(f'Frames co detected movement: {n_detected}/{len(frames)} ({100*n_detected/len(frames):.1f}%)')

# ── Cumulative + pitch offsets ─────────────────────────────────────────────
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from utils.stub_io import load_track_stub
import os, json

cum = CameraMovementEstimator.cumulative(camera_movement)
cum_arr = np.array(cum)
cum_x_span = float(cum_arr[:, 0].max() - cum_arr[:, 0].min())

vt      = ViewTransformer(frame_size=(frames[0].shape[1], frames[0].shape[0]))
mpp     = vt.pan_scale_mpp()
offsets = vt.compute_pitch_offsets(cum, pitch_x_start=0.0)
off_span = max(offsets) - min(offsets)

print()
print(f'm_per_px    : {mpp:.4f}')
print(f'cum_x span  : {cum_x_span:.0f} px')
print(f'offset span : {off_span:.1f} m')

# ── Apply to tracking stub ────────────────────────────────────────────────
# Find latest stub matching this video
import json as _json
latest_stub = None
for jdir in sorted(os.listdir('output_videos'), reverse=True):
    meta_f = os.path.join('output_videos', jdir, 'job_meta.json')
    if not os.path.exists(meta_f):
        continue
    meta = _json.load(open(meta_f))
    inp  = meta.get('input_path', '')
    stub_f = os.path.join('stubs', jdir + '_track_stubs.pkl')
    if os.path.exists(stub_f):
        latest_stub = stub_f
        print(f'Track stub  : {stub_f}')
        break

if latest_stub:
    tracks, _, _ = load_track_stub(latest_stub)
    if len(tracks['players']) == len(frames):
        # add position_adjusted from raw position
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
                    all_x.append(pt[0]); all_y.append(pt[1])
        if all_x:
            x_span = max(all_x) - min(all_x)
            y_span = max(all_y) - min(all_y)
            print()
            print(f'[Pitch] Kich thuoc san do duoc (CODE MOI):')
            print(f'  dai  = {x_span:.1f} m   ({min(all_x):.1f} -> {max(all_x):.1f} m)')
            print(f'  rong = {y_span:.1f} m   ({min(all_y):.1f} -> {max(all_y):.1f} m)')
            if x_span >= 80:
                print('=> [OK] chiều dài hợp lý')
            elif x_span >= 50:
                print('=> [WARN] chieu dai trung binh (50-80m)')
            else:
                print('=> [FAIL] chieu dai qua nho (<50m)')
    else:
        print(f'Stub co {len(tracks["players"])} frames, video co {len(frames)} frames — khong khop, bo qua transform')
else:
    print('Khong tim thay tracking stub phu hop')
    print(f'Uoc tinh chieu dai san: {23.32 + off_span:.1f} m  (23.32 m visible + {off_span:.1f} m lia)')
