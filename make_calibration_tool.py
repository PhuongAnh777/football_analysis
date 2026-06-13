"""Tạo file HTML cho phép click chuột để lấy tọa độ pixel trên frame sân."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import cv2, base64, numpy as np, os

VIDEO = sys.argv[1] if len(sys.argv) > 1 else 'input_videos/input_video.mp4'

cap = cv2.VideoCapture(VIDEO)
ret, frame = cap.read()
cap.release()
h, w = frame.shape[:2]
print(f'Frame: {w}x{h}')

# Ve luoi len anh
out = frame.copy()
for y in range(0, h, 50):
    cv2.line(out, (0, y), (w, y), (180, 180, 180), 1)
    cv2.putText(out, str(y), (2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 220), 1)
for x in range(0, w, 50):
    cv2.line(out, (x, 0), (x, h), (180, 180, 180), 1)
    cv2.putText(out, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 220), 1)

# Encode base64
_, buf = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
b64 = base64.b64encode(buf.tobytes()).decode()

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Pitch Calibration Tool</title>
<style>
body {{ background:#111; color:#eee; font-family:monospace; margin:10px; }}
canvas {{ cursor:crosshair; border:1px solid #555; display:block; }}
#info {{ margin:8px 0; font-size:14px; }}
#pts {{ margin:8px 0; font-size:13px; line-height:1.7; }}
.pt-done {{ color:#4ade80; }}
.pt-next {{ color:#fbbf24; font-weight:bold; }}
.pt-wait {{ color:#666; }}
button {{ background:#374151; color:#eee; border:1px solid #555; padding:6px 16px;
          margin:4px; cursor:pointer; border-radius:4px; font-size:13px; }}
button:hover {{ background:#4b5563; }}
#result {{ background:#1f2937; padding:12px; border-radius:6px; margin-top:10px;
           white-space:pre; font-size:12px; display:none; }}
</style>
</head>
<body>
<h3 style="margin:4px 0">Pitch Calibration — click chọn 4 điểm trên sân</h3>
<div id="info">Hover chuột: <b id="coords">-</b></div>
<div id="pts">
  <span id="p0" class="pt-next">① near-left  → goal line + biên GẦN (phía dưới ảnh)</span><br>
  <span id="p1" class="pt-wait">② far-left   → goal line + biên XA  (phía trên ảnh)</span><br>
  <span id="p2" class="pt-wait">③ far-right  → vạch phải + biên XA</span><br>
  <span id="p3" class="pt-wait">④ near-right → vạch phải + biên GẦN</span>
</div>
<canvas id="cv" width="{w}" height="{h}"></canvas>
<div style="margin-top:8px">
  <button onclick="undo()">↩ Undo</button>
  <button onclick="reset()">🔄 Reset</button>
  <button onclick="generate()" id="genBtn" disabled>✅ Tạo code calibration</button>
</div>
<div style="margin-top:8px">
  x_max (mét) = <input id="xmax" type="number" value="88.5" step="0.5" style="width:70px;background:#374151;color:#eee;border:1px solid #555;padding:3px">
  <small style="color:#9ca3af"> (52.5=vạch giữa, 88.5=penalty xa, 105=cổng phải)</small>
</div>
<div id="result"></div>

<script>
const img = new Image();
img.src = 'data:image/jpeg;base64,{b64}';
const cv  = document.getElementById('cv');
const ctx = cv.getContext('2d');
const W = {w}, H = {h};
let pts = [];
const LABELS = ['near-left(0,68)','far-left(0,0)','far-right(xmax,0)','near-right(xmax,68)'];
const COLORS = ['#60a5fa','#34d399','#f87171','#fbbf24'];

img.onload = () => {{ ctx.drawImage(img, 0, 0); }};

cv.addEventListener('mousemove', e => {{
  const r = cv.getBoundingClientRect();
  const x = Math.round(e.clientX - r.left);
  const y = Math.round(e.clientY - r.top);
  document.getElementById('coords').textContent = `x=${{x}}, y=${{y}}`;
}});

cv.addEventListener('click', e => {{
  if (pts.length >= 4) return;
  const r = cv.getBoundingClientRect();
  const x = Math.round(e.clientX - r.left);
  const y = Math.round(e.clientY - r.top);
  pts.push([x, y]);
  redraw();
  updateLabels();
  if (pts.length === 4) document.getElementById('genBtn').disabled = false;
}});

function redraw() {{
  ctx.drawImage(img, 0, 0);
  pts.forEach(([x,y], i) => {{
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI*2);
    ctx.fillStyle = COLORS[i];
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 13px monospace';
    ctx.fillText(i+1, x+10, y-6);
    ctx.fillStyle = COLORS[i];
    ctx.font = '11px monospace';
    ctx.fillText(`(${{x}},${{y}})`, x+10, y+8);
  }});
  if (pts.length >= 2) {{
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    pts.forEach(([x,y]) => ctx.lineTo(x, y));
    if (pts.length === 4) ctx.closePath();
    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6,4]);
    ctx.stroke();
    ctx.setLineDash([]);
  }}
}}

function updateLabels() {{
  for (let i = 0; i < 4; i++) {{
    const el = document.getElementById('p'+i);
    if (i < pts.length) el.className = 'pt-done';
    else if (i === pts.length) el.className = 'pt-next';
    else el.className = 'pt-wait';
  }}
}}

function undo() {{
  pts.pop();
  document.getElementById('genBtn').disabled = pts.length < 4;
  redraw(); updateLabels();
}}

function reset() {{
  pts = [];
  document.getElementById('genBtn').disabled = true;
  redraw(); updateLabels();
  document.getElementById('result').style.display = 'none';
}}

function generate() {{
  const xmax = parseFloat(document.getElementById('xmax').value) || 88.5;
  const [nl, fl, fr, nr] = pts;
  const code = `    # Calibrated for {w}x{h} video (recalibrated)
    # near-left  ${{nl}} → (x=0,     y=68)
    # far-left   ${{fl}} → (x=0,     y=0 )
    # far-right  ${{fr}} → (x=${{xmax}}, y=0 )
    # near-right ${{nr}} → (x=${{xmax}}, y=68)
    _DEFAULT_PIXELS_1920: np.ndarray = np.array([
        [${{nl[0]}},  ${{nl[1]}}],
        [${{fl[0]}},   ${{fl[1]}}],
        [${{fr[0]}},   ${{fr[1]}}],
        [${{nr[0]}},  ${{nr[1]}}],
    ], dtype=np.float32)

    _DEFAULT_REGION: PitchRegion = PitchRegion(
        x_min=0.0, x_max=${{xmax}}, y_min=0.0, y_max=68.0
    )

    _REF_W: int = {w}
    _REF_H: int = {h}`;

  const el = document.getElementById('result');
  el.textContent = code;
  el.style.display = 'block';
}}
</script>
</body>
</html>"""

out_path = 'calibration_tool.html'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f'Saved: {out_path}')
print(f'Mo file nay trong browser: {os.path.abspath(out_path)}')
