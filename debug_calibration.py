import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import cv2, numpy as np

cap = cv2.VideoCapture('input_videos/input_video.mp4')
ret, frame = cap.read()
cap.release()
h, w = frame.shape[:2]
print(f'Frame size: {w}x{h}')

# Calibration goc thiet ke cho 1920x1080
orig = [(110, 1035), (265, 275), (910, 260), (1640, 915)]
# Scale xuong 1280x720
sx, sy = w / 1920, h / 1080
scaled = [(int(x * sx), int(y * sy)) for x, y in orig]
labels = ['near-left (0m,y=68)', 'far-left (0m,y=0)', 'far-right (23m,y=0)', 'near-right (23m,y=68)']
print('Scaled calibration corners:')
for lbl, o, s in zip(labels, orig, scaled):
    inside = 0 <= s[0] < w and 0 <= s[1] < h
    status = 'OK' if inside else 'NGOAI FRAME'
    print(f'  {lbl}: {o} -> {s}  [{status}]')

frame2 = frame.copy()
# Do: goc cu sai
pts_orig = np.array([[min(x, w-1), min(y, h-1)] for x, y in orig], np.int32)
cv2.polylines(frame2, [pts_orig], True, (0, 0, 255), 2)
cv2.putText(frame2, 'SAI: 1920x1080', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Xanh: scaled dung
pts_new = np.array(scaled, np.int32)
cv2.polylines(frame2, [pts_new], True, (0, 255, 0), 3)
cv2.putText(frame2, 'SCALED: 1280x720', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
for p, lbl in zip(scaled, labels):
    cv2.circle(frame2, p, 8, (0, 255, 0), -1)
    cv2.putText(frame2, lbl, (p[0]+5, p[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

cv2.imwrite('debug_frame0_scaled.jpg', frame2)
print('Saved: debug_frame0_scaled.jpg')
