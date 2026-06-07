"""
TacticalNarrator
================
Calls an LLM (OpenAI-compatible endpoint) to produce a structured,
data-grounded Vietnamese tactical analysis from a ``ReportBuilder``
``match_report`` dict.

Enforced rules (baked into the system prompt)
----------------------------------------------
1. Every claim MUST cite a specific number from the provided JSON.
2. All comparisons are within-match — no external benchmarks.
3. Players referenced as "Cầu thủ #<track_id>".
4. Professional magazine-style tone — not live commentary.
5. Output is pure JSON matching ``OUTPUT_SCHEMA`` exactly.
6. Every item in diem_manh / diem_yeu must contain ≥1 number.
7. Defensive line & team width: neutral style description, no good/bad.
8. xep_loai grade must be consistent with diem_so_tong in the JSON.
9. turnovers_final_third: use tactical framing, never "tệ/yếu kém".
10. passing_score = None: write "Không đủ dữ liệu chuyền bóng để đánh giá."
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
import urllib.error
from typing import Any

# ── canonical output schema ──────────────────────────────────────────────────

OUTPUT_SCHEMA: dict[str, Any] = {
    "tong_quan_tran_dau": {
        "nhan_xet_chung": "<3-4 câu tổng quan, đề cập tactical_profile cả 2 đội>",
        "doi_noi_bat":    "<1, 2, hoặc null>",
        "ly_do":          "<1-2 câu lý giải dựa trên số liệu cụ thể>",
    },
    "danh_gia_doi": {
        "doi_1": {
            "diem_so_tong":        "<overall_score từ JSON, kiểu float>",
            "xep_loai":            "<A|B|C|D|F — phải nhất quán với diem_so_tong>",
            "tactical_profile":    "<lấy từ JSON, mô tả thêm 1 câu giải thích>",
            "chien_thuat":         "<sơ đồ + cách chơi, 2-3 câu có số liệu>",
            "diem_manh":           ["<2-3 điểm mạnh, mỗi điểm BẮT BUỘC có con số>"],
            "diem_yeu":            ["<2-3 điểm yếu, mỗi điểm BẮT BUỘC có con số>"],
            "nhan_xet_pressing":   "<1-2 câu: intensity trung bình, hiệp cao hơn, so với đội kia>",
            "nhan_xet_doi_hinh":   "<compact score + adherence score + broken rate>",
            "nhan_xet_toc_do":     "<speed km/h + sprint % + so sánh với đội kia>",
            "nhan_xet_hang_thu":   "<def_line_height avg, style (high/mid/low block), trend — TRUNG LẬP>",
            "nhan_xet_do_rong":    "<width avg, khi có/không bóng, style — TRUNG LẬP>",
            "nhan_xet_van_dong":   "<high intensity runs: tổng số, phân bố DEF/MID/FWD, 1-2 câu>",
            "nhan_xet_tranh_chap": "<ball recoveries: tổng số, vùng cướp bóng, 1-2 câu>",
            "nhan_xet_mat_bong":   "<turnovers final third: số lần, dangerous_rate — KHÔNG dùng 'tệ/yếu kém'>",
            "nhan_xet_chuyen":     "<passing stats nếu có, hoặc 'Không đủ dữ liệu chuyền bóng để đánh giá chỉ số này.'>",
        },
        "doi_2": {
            "diem_so_tong":        "<overall_score từ JSON, kiểu float>",
            "xep_loai":            "<A|B|C|D|F>",
            "tactical_profile":    "<lấy từ JSON, mô tả thêm 1 câu>",
            "chien_thuat":         "<sơ đồ + cách chơi, 2-3 câu>",
            "diem_manh":           ["<2-3 điểm mạnh có con số>"],
            "diem_yeu":            ["<2-3 điểm yếu có con số>"],
            "nhan_xet_pressing":   "<1-2 câu>",
            "nhan_xet_doi_hinh":   "<compact + adherence + broken rate>",
            "nhan_xet_toc_do":     "<speed + sprint % + so sánh>",
            "nhan_xet_hang_thu":   "<def_line_height avg, style, trend — TRUNG LẬP>",
            "nhan_xet_do_rong":    "<width avg, khi có/không bóng — TRUNG LẬP>",
            "nhan_xet_van_dong":   "<high intensity runs theo vị trí, 1-2 câu>",
            "nhan_xet_tranh_chap": "<ball recoveries + vùng cướp bóng, 1-2 câu>",
            "nhan_xet_mat_bong":   "<turnovers final third + dangerous_rate — KHÔNG dùng 'tệ/yếu kém'>",
            "nhan_xet_chuyen":     "<passing stats hoặc 'Không đủ dữ liệu chuyền bóng để đánh giá chỉ số này.'>",
        },
    },
    "danh_gia_cau_thu": {
        "doi_1": {
            "xuat_sac": {
                "track_id":       "<int — track_id từ top_players[0]>",
                "ly_do":          "<1-2 câu có số liệu: overall_score, grade, strengths>",
                "chi_so_noi_bat": "<tên metric và giá trị nổi bật nhất>",
            },
            "can_cai_thien": {
                "track_id":    "<int — track_id từ weak_players[0]>",
                "ly_do":       "<1-2 câu có số liệu: overall_score, grade, weaknesses>",
                "khuyen_nghi": "<1 câu khuyến nghị cải thiện cụ thể>",
            },
        },
        "doi_2": {
            "xuat_sac":      {"track_id": "<int>", "ly_do": "<str>", "chi_so_noi_bat": "<str>"},
            "can_cai_thien": {"track_id": "<int>", "ly_do": "<str>", "khuyen_nghi":    "<str>"},
        },
    },
    "so_sanh_doi_dau": {
        "pressing":             "<ai pressing hiệu quả hơn, con số cụ thể của cả 2 đội>",
        "doi_hinh":             "<kỷ luật chiến thuật: compact + adherence, so sánh 2 đội>",
        "the_luc":              "<tốc độ km/h + sprint % cả 2 đội, ai vượt trội>",
        "kiem_soat_bong":       "<possession % cả 2 đội, nhận xét chất lượng kiểm soát>",
        "chien_luoc_phong_ngu": "<so sánh def_line avg cả 2 đội, ý đồ chiến thuật — TRUNG LẬP>",
        "su_dung_bien":         "<so sánh width avg + behavior khi có/không bóng — TRUNG LẬP>",
        "van_dong":             "<so sánh high intensity runs: tổng số + phân bố DEF/MID/FWD cả 2 đội>",
        "tranh_chap":           "<so sánh ball recoveries: tổng số + vùng sân cả 2 đội>",
        "mat_bong":             "<ai mất bóng ở final third nhiều hơn, dangerous_rate — không phán xét>",
        "kien_tao":             "<so sánh passing stats nếu có, hoặc bỏ qua nếu cả 2 đội đều None>",
    },
    "ket_luan": "<2-3 câu tổng kết, đề cập tactical profile cả 2 đội, không lặp số liệu đã nêu>",
}

# ── required keys for validation ────────────────────────────────────────────

_REQUIRED_TOP    = {"tong_quan_tran_dau", "danh_gia_doi", "danh_gia_cau_thu",
                    "so_sanh_doi_dau",    "ket_luan"}
_REQUIRED_TEAM   = {
    "diem_so_tong", "xep_loai", "tactical_profile", "chien_thuat",
    "diem_manh",    "diem_yeu",  "nhan_xet_pressing",
    "nhan_xet_doi_hinh", "nhan_xet_toc_do",
    "nhan_xet_hang_thu", "nhan_xet_do_rong",
    "nhan_xet_van_dong", "nhan_xet_tranh_chap",
    "nhan_xet_mat_bong", "nhan_xet_chuyen",
}
_REQUIRED_PLAYER = {"track_id", "ly_do"}
_VALID_GRADES    = {"A", "B", "C", "D", "F"}

_GRADE_THRESHOLDS = ((80, "A"), (65, "B"), (50, "C"), (35, "D"))


def _infer_grade(score: float) -> str:
    for threshold, letter in _GRADE_THRESHOLDS:
        if score >= threshold:
            return letter
    return "F"


# ── main class ───────────────────────────────────────────────────────────────

class TacticalNarrator:
    """Generate a structured Vietnamese tactical analysis via an LLM.

    Parameters
    ----------
    api_key : str
        API key for the LLM provider.
    model : str
        Model identifier (default ``"gpt-4o"``).
    base_url : str
        OpenAI-compatible chat completions base URL.
    temperature : float
        Sampling temperature (default 0.3).
    max_tokens : int
        Max completion tokens (default 8192).
    timeout : int
        HTTP timeout in seconds (default 180).
    max_retries : int
        Retry attempts on transient errors (default 3).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.3,
        max_tokens: int = 8192,
        timeout: int = 180,
        max_retries: int = 3,
    ) -> None:
        self.api_key     = api_key
        self.model       = model
        self.base_url    = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.timeout     = timeout
        self.max_retries = max_retries

    # ── public API ───────────────────────────────────────────────────────────

    def analyze(self, match_report: dict) -> dict[str, Any]:
        """Generate structured tactical analysis from a ``match_report``.

        Returns
        -------
        JSON-serialisable dict matching ``OUTPUT_SCHEMA``.

        Raises
        ------
        RuntimeError
            If all retry attempts fail or the response cannot be parsed.
        """
        messages  = self._build_messages(match_report)
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                token_budget = min(self.max_tokens * (2 ** (attempt - 1)), 16384)
                raw, finish_reason = self._call_api(messages, max_tokens=token_budget)
                if finish_reason == "length":
                    raise ValueError(
                        f"Response truncated (finish_reason=length, max_tokens={token_budget})"
                    )
                return self._parse_and_validate(raw)
            except (ValueError, KeyError) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    messages = self._build_messages(match_report)
                    messages.append({
                        "role": "user",
                        "content": (
                            "Phản hồi trước bị cắt hoặc JSON không hợp lệ. "
                            "Trả về JSON đầy đủ theo schema, đóng ngoặc hoàn chỉnh. "
                            "Mỗi nhận xét 1-2 câu ngắn gọn nhưng vẫn có số liệu."
                        ),
                    })
                    time.sleep(2 ** attempt)

        raise RuntimeError(
            f"TacticalNarrator: all {self.max_retries} attempts failed. "
            f"Last error: {last_exc}"
        )

    def build_prompt(self, match_report: dict) -> list[dict]:
        """Return the messages list without making an API call."""
        return self._build_messages(match_report)

    # ── prompt construction ──────────────────────────────────────────────────

    def _build_messages(self, match_report: dict) -> list[dict]:
        return [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user",   "content": self._user_prompt(match_report)},
        ]

    def _system_prompt(self) -> str:
        schema_str = json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, indent=2)
        return f"""\
Bạn là chuyên gia phân tích chiến thuật bóng đá với 15 năm kinh nghiệm,
từng làm việc cho các đội bóng chuyên nghiệp.
Phong cách viết: rõ ràng, chiều sâu chiến thuật, như bài phân tích tạp chí,
KHÔNG phải tường thuật trực tiếp.

════════════════════ QUY TẮC BẮT BUỘC ════════════════════

RULE 1 — LUÔN CÓ SỐ LIỆU
Mỗi nhận xét phải trích dẫn ít nhất một con số từ JSON.
✗ "Đội 1 pressing tốt."
✓ "Đội 1 duy trì pressing 2.8 cầu thủ/bán kính 8m, cao hơn Đội 2 (1.6),
   đặc biệt hiệp 2 tăng đáng kể từ 2.1 lên 3.8."

RULE 2 — SO SÁNH NỘI BỘ
Chỉ so sánh 2 đội với nhau trong trận này. Không dùng chuẩn ngoài.

RULE 3 — TÊN CẦU THỦ
Gọi là "Cầu thủ #<track_id>". Không đặt tên.

RULE 4 — PHONG CÁCH
Chuyên nghiệp, dễ đọc. Tránh ngôn ngữ AI chung như "đáng chú ý",
"điều này cho thấy", "cần lưu ý". Viết trực tiếp.

RULE 5 — OUTPUT THUẦN JSON
Trả về JSON thuần túy theo đúng schema. Không có markdown, không giải thích.

RULE 6 — SỐ LIỆU TRONG ĐIỂM MẠNH/YẾU
Mỗi phần tử trong diem_manh và diem_yeu bắt buộc có ít nhất 1 con số.

RULE 7 — TRUNG LẬP VỀ HÀNG THỦ VÀ ĐỘ RỘNG
Không phán xét high_block/low_block hay wide/narrow là tốt/xấu.
Mô tả chiến thuật đằng sau số liệu.
✗ "Hàng thủ thấp 6.2m cho thấy đội yếu, bị động."
✓ "Đội 2 duy trì hàng thủ trung bình 6.2m (low block), thể hiện chủ trương
   phòng thủ compact, nhường thế trận và chờ cơ hội phản công."

RULE 8 — NHẤT QUÁN GRADE
xep_loai phải khớp với diem_so_tong: A≥80 | B≥65 | C≥50 | D≥35 | F<35.

RULE 9 — TURNOVERS: DIỄN GIẢI CHIẾN THUẬT
Với turnovers_final_third: KHÔNG dùng "tệ" hay "yếu kém".
✗ "Đội 1 mất bóng nhiều ở 1/3 cuối sân, rất tệ."
✓ "Đội 1 chấp nhận rủi ro cao khi kiến tạo từ sâu, với 7 lần mất bóng
   ở 1/3 cuối sân (dangerous_rate 45%)."

RULE 10 — THIẾU DỮ LIỆU CHUYỀN BÓNG
Nếu passing_score = null hoặc total_passes = null: ghi chính xác
"Không đủ dữ liệu chuyền bóng để đánh giá chỉ số này." — không bịa số.

════════════════════ OUTPUT SCHEMA ════════════════════

{schema_str}

Lưu ý quan trọng:
- "doi_noi_bat" là số nguyên 1, 2, hoặc null (không phải chuỗi).
- "diem_manh" và "diem_yeu" mỗi mảng tối thiểu 2 phần tử.
- "xuat_sac.track_id" và "can_cai_thien.track_id" là số nguyên.
- Nhận xét cầu thủ chỉ dùng cầu thủ có trong top_players / weak_players.
- nhan_xet_hang_thu và nhan_xet_do_rong: TUYỆT ĐỐI trung lập.
- nhan_xet_mat_bong: KHÔNG dùng "tệ" hay "yếu kém" — chỉ diễn giải chiến thuật.
- nhan_xet_chuyen: nếu thiếu dữ liệu → ghi đúng câu quy định ở RULE 10.
- so_sanh_doi_dau.kien_tao: nếu cả 2 đội không có passing data → ghi "Không đủ dữ liệu."
"""

    def _user_prompt(self, match_report: dict) -> str:
        report_json = json.dumps(match_report, ensure_ascii=False, indent=2)
        return f"""\
Dưới đây là báo cáo thống kê trận đấu được trích xuất từ video phân tích:

<match_report>
{report_json}
</match_report>

Hãy viết báo cáo đánh giá theo ĐÚNG cấu trúc JSON đã mô tả trong system prompt.
Trả về JSON thuần túy, không có markdown, không có giải thích ngoài.
"""

    # ── API call ─────────────────────────────────────────────────────────────

    def _call_api(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
    ) -> tuple[str, str | None]:
        url     = f"{self.base_url}/chat/completions"
        payload = json.dumps({
            "model":       self.model,
            "messages":    messages,
            "temperature": self.temperature,
            "max_tokens":  max_tokens if max_tokens is not None else self.max_tokens,
            "response_format": {"type": "json_object"},
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data    = payload,
            method  = "POST",
            headers = {
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ValueError(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}") from exc

        choice = body["choices"][0]
        return choice["message"]["content"], choice.get("finish_reason")

    # ── parse & validate ─────────────────────────────────────────────────────

    def _parse_and_validate(self, raw: str) -> dict[str, Any]:
        # Strip markdown fences if present
        stripped = raw.strip()
        fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", stripped)
        if fence_match:
            stripped = fence_match.group(1).strip()

        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned invalid JSON: {exc}\nRaw: {raw[:400]}") from exc

        # Top-level keys
        missing_top = _REQUIRED_TOP - set(data.keys())
        if missing_top:
            raise KeyError(f"Missing top-level keys: {missing_top}")

        # Team section
        for doi_key in ("doi_1", "doi_2"):
            team_data = data.get("danh_gia_doi", {}).get(doi_key, {})
            missing   = _REQUIRED_TEAM - set(team_data.keys())
            if missing:
                raise KeyError(f"danh_gia_doi.{doi_key} missing keys: {missing}")
            # Grade consistency
            grade = team_data.get("xep_loai", "")
            if grade not in _VALID_GRADES:
                try:
                    score = float(team_data.get("diem_so_tong", 50))
                    team_data["xep_loai"] = _infer_grade(score)
                except (TypeError, ValueError):
                    team_data["xep_loai"] = "C"

        # Player section
        for doi_key in ("doi_1", "doi_2"):
            for role_key in ("xuat_sac", "can_cai_thien"):
                player_data = data.get("danh_gia_cau_thu", {}).get(doi_key, {}).get(role_key, {})
                missing     = _REQUIRED_PLAYER - set(player_data.keys())
                if missing:
                    raise KeyError(
                        f"danh_gia_cau_thu.{doi_key}.{role_key} missing keys: {missing}"
                    )
                # Coerce track_id to int
                try:
                    player_data["track_id"] = int(player_data["track_id"])
                except (TypeError, ValueError):
                    player_data["track_id"] = 0

        return data
