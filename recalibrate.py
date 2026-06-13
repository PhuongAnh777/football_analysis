"""
recalibrate.py
==============
Nhấn chuột chọn 4 điểm trên sân để tạo calibration mới cho ViewTransformer.

Cách dùng
---------
    python recalibrate.py [video_path]

Bước 1: Cửa sổ ảnh sẽ hiện lên với các vạch sân được đánh dấu gợi ý.
Bước 2: Chọn đúng 4 điểm theo thứ tự:
    1. Góc TRÁI GẦN  (near-left):  Điểm giao vạch GÓN BÊN TRÁI + đường BIÊN GẦN  (camera side)
    2. Góc TRÁI XA   (far-left):   Điểm giao vạch GÓN BÊN TRÁI + đường BIÊN XA   (fan side)
    3. Góc PHẢI XA   (far-right):  Điểm giao vạch PHẢI         + đường BIÊN XA
    4. Góc PHẢI GẦN  (near-right): Điểm giao vạch PHẢI         + đường BIÊN GẦN

    "Vạch phải" có thể là: vạch giữa sân (52.5m), vạch penalty xa (88.5m), hoặc khung thành phải
    Quan trọng: 4 điểm phải là 4 góc của 1 hình CHỮ NHẬT thực trên sân.

Bước 3: Nhấn S để lưu, ESC để huỷ.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import cv2, numpy as np

VIDEO = sys.argv[1] if len(sys.argv) > 1 else 'input_videos/input_video.mp4'

cap = cv2.VideoCapture(VIDEO)
ret, frame = cap.read()
cap.release()
if not ret:
    print(f'Khong the doc video: {VIDEO}')
    sys.exit(1)

h, w = frame.shape[:2]
print(f'Video: {w}x{h}')

# Goi y: ve calibration cu (scaled) de tham khao
sx, sy = w / 1920, h / 1080
old_pts = np.array([
    [int(110*sx), int(1035*sy)],
    [int(265*sx), int(275*sy)],
    [int(910*sx), int(260*sy)],
    [int(1640*sx), int(915*sy)],
], np.int32)
frame_ref = frame.copy()
cv2.polylines(frame_ref, [old_pts], True, (0, 0, 255), 1)
cv2.putText(frame_ref, 'Calibration cu (SAI)', (10, h-20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

# Ve goi y cac diem nen chon
HINTS = [
    ((int(w*0.06), int(h*0.93)), 'near-left (0m, near)'),
    ((int(w*0.14), int(h*0.26)), 'far-left  (0m, far)'),
    ((int(w*0.87), int(h*0.26)), 'far-right (Xm, far)'),
    ((int(w*0.86), int(h*0.83)), 'near-right(Xm, near)'),
]
for (x,y), lbl in HINTS:
    cv2.circle(frame_ref, (x,y), 8, (255, 165, 0), 2)
    cv2.putText(frame_ref, lbl, (x+8, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,165,0), 1)

LABELS = [
    'near-left  → goal line, near touchline  (x=0, y=68)',
    'far-left   → goal line, far touchline   (x=0, y=0)',
    'far-right  → right boundary, far touch  (x=?, y=0)',
    'near-right → right boundary, near touch (x=?, y=68)',
]

selected = []
display  = frame_ref.copy()

def mouse_cb(event, mx, my, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(selected) < 4:
        selected.append((mx, my))
        idx = len(selected) - 1
        cv2.circle(display, (mx, my), 10, (0,255,0), -1)
        cv2.putText(display, str(idx+1), (mx+8, my-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        if len(selected) > 1:
            cv2.line(display, selected[-2], selected[-1], (0,255,0), 2)
        if len(selected) == 4:
            cv2.line(display, selected[-1], selected[0], (0,255,0), 2)
        cv2.imshow('Calibration', display)
        print(f'  [{idx+1}] pixel ({mx}, {my})  ← {LABELS[idx]}')

cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Calibration', min(w, 1280), min(h, 720))
cv2.setMouseCallback('Calibration', mouse_cb)
cv2.imshow('Calibration', display)

print()
print('Nhan chuot chon 4 diem theo thu tu:')
for i, lbl in enumerate(LABELS):
    print(f'  {i+1}. {lbl}')
print()
print('  S = luu calibration')
print('  R = reset (chon lai tu dau)')
print('  ESC = thoat')
print()

x_max_world = None

while True:
    key = cv2.waitKey(0) & 0xFF
    if key == 27:   # ESC
        print('Huy bo.')
        break
    elif key == ord('r') or key == ord('R'):
        selected.clear()
        display[:] = frame_ref[:]
        cv2.imshow('Calibration', display)
        print('Reset — chon lai 4 diem:')
    elif key == ord('s') or key == ord('S'):
        if len(selected) != 4:
            print(f'Can chon du 4 diem (hien co {len(selected)})')
            continue

        # Hoi x_max
        print()
        x_max_str = input('Nhap x_max (met) tuong ung voi diem phai (vi du: 52.5 hoac 88.5 hoac 105): ').strip()
        try:
            x_max_world = float(x_max_str)
        except ValueError:
            print('Gia tri khong hop le, dung 88.5m')
            x_max_world = 88.5

        pv = np.array(selected, dtype=np.float32)
        print()
        print('=' * 55)
        print('COPY DOAN NAY VAO view_transformer/view_transformer.py:')
        print('=' * 55)
        print(f"""
    # Calibrated for {w}x{h} video
    # near-left  {selected[0]} → (x=0,       y=68)
    # far-left   {selected[1]} → (x=0,       y=0 )
    # far-right  {selected[2]} → (x={x_max_world}, y=0 )
    # near-right {selected[3]} → (x={x_max_world}, y=68)
    _DEFAULT_PIXELS_1920: np.ndarray = np.array([
        [{selected[0][0]},  {selected[0][1]}],
        [{selected[1][0]},   {selected[1][1]}],
        [{selected[2][0]},   {selected[2][1]}],
        [{selected[3][0]},  {selected[3][1]}],
    ], dtype=np.float32)

    _DEFAULT_REGION: PitchRegion = PitchRegion(
        x_min=0.0, x_max={x_max_world}, y_min=0.0, y_max=68.0
    )

    _REF_W: int = {w}
    _REF_H: int = {h}
""")
        print('=' * 55)

        # Verify
        from view_transformer import ViewTransformer, PitchRegion
        pr = PitchRegion(x_min=0.0, x_max=x_max_world, y_min=0.0, y_max=68.0)
        vt = ViewTransformer(pixel_vertices=pv, pitch_region=pr)
        pts_test = [
            (selected[0], '(0,68)'),
            (selected[1], '(0,0)'),
            (selected[2], f'({x_max_world},0)'),
            (selected[3], f'({x_max_world},68)'),
        ]
        print('Kiem tra transform:')
        for (px, py), expected in pts_test:
            result = vt.transform_point(np.array([px, py], dtype=np.float32))
            if result is not None:
                rx, ry = result.squeeze()
                print(f'  pixel ({px},{py}) → ({rx:.1f}, {ry:.1f})  ky vong {expected}')
            else:
                print(f'  pixel ({px},{py}) → None  [NGOAI polygon]')
        break

cv2.destroyAllWindows()
