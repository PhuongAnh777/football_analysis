# Football Analysis

Hệ thống phân tích chiến thuật bóng đá từ video broadcast, sử dụng YOLO + BoT-SORT tracking, ước lượng camera movement, gán đội, và engine phân tích chiến thuật.

## Cấu trúc

| Thành phần | Mô tả |
|---|---|
| `main.py` | Pipeline CLI (dùng stub cache cho dev nhanh) |
| `api/` | FastAPI REST API |
| `frontend/` | React dashboard (Vite + Tailwind) |
| `app.py` | Streamlit UI (tùy chọn) |
| `trackers/` | YOLO detection + BoT-SORT + ReID + track merger |
| `tactical_analyzer/` | Phân tích chiến thuật, scoring, báo cáo, LLM narrator |

## Cài đặt

```bash
pip install -r requirements_api.txt
cd frontend && npm install
```

Model YOLO custom: `models/best.pt` (đã có trong repo).

## Chạy (React + API — khuyến nghị)

**Terminal 1 — Backend:**
```bash
uvicorn api.main_api:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

Mở http://localhost:3000 → Upload video MP4/AVI (≤ 500 MB) → Dashboard.

### Báo cáo AI (tùy chọn)

```bash
set OPENAI_API_KEY=sk-...
set LLM_MODEL=gpt-4o
```

Không có API key vẫn chạy được — dashboard hiển thị số liệu từ `match_report`, báo cáo văn bản dùng template fallback.

## Chạy CLI

```bash
python main.py
```

Dùng `input_videos/input_video.mp4` và stub cache trong `stubs/` (bật `read_from_stub=True`).

## Chạy Streamlit

```bash
streamlit run app.py
```

## API Endpoints

| Method | Path | Mô tả |
|---|---|---|
| POST | `/api/analyze` | Upload video (field: `video`) |
| GET | `/api/status/{job_id}` | Tiến trình |
| GET | `/api/results/{job_id}` | Kết quả đầy đủ |
| GET | `/api/video/{job_id}` | Stream video output |
| GET | `/api/health` | Health check |

## Output

Kết quả lưu tại `output_videos/`:
- `output_video.avi` / `.mp4` — video annotated
- `heatmap_team1/2.png`, `passing_network_team1/2.png`
- `tactical_report.json`, `scored_report.json`, `match_report.json`
- `llm_analysis.json` (nếu có OPENAI_API_KEY)
