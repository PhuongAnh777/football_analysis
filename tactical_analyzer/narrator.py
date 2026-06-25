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
4. Professional, concise tone — coaching debrief style, not live commentary.
5. Output is pure JSON matching ``OUTPUT_SCHEMA`` exactly.
6. Every item in diem_manh / diem_yeu must contain ≥1 number.
7. Defensive line & team width: neutral style description, no good/bad.
8. turnovers_final_third: use tactical framing, never "tệ/yếu kém".
9. No overall scores or letter grades in output.
10. No passing data → write "Không đủ dữ liệu chuyền bóng."
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
        "nhan_xet_chung": "<3-4 câu tổng quan mô hình chiến thuật cả 2 đội>",
        "doi_noi_bat":    "<1, 2, hoặc null — số nguyên>",
        "ly_do":          "<1-2 câu lý giải dựa trên số liệu cụ thể>",
    },
    "danh_gia_doi": {
        "doi_1": {
            "chien_thuat":         "<sơ đồ + block style + cách chơi, 2-3 câu có số liệu>",
            "pressing":            "<PPDA nếu có, hoặc proximity; nửa đầu/nửa sau video, 1-2 câu>",
            "doi_hinh":            "<compact theo phase tấn công/phòng ngự (m²) + width delta, 1-2 câu — trung lập>",
            "hang_thu":            "<def_line_height avg, block style, trend — TUYỆT ĐỐI TRUNG LẬP>",
            "xay_dung":            "<progressive_pass_pct + high_risk_count/total + avg_distance_to_goal, 1-2 câu>",
            "diem_manh":           ["<2-3 điểm mạnh, mỗi điểm BẮT BUỘC có con số>"],
            "diem_yeu":            ["<2-3 điểm yếu, mỗi điểm BẮT BUỘC có con số>"],
            "khuyen_nghi_hlv":     ["<2 khuyến nghị hành động cụ thể cho ban huấn luyện>"],
        },
        "doi_2": {
            "chien_thuat":         "<sơ đồ + block style + cách chơi, 2-3 câu>",
            "pressing":            "<nửa đầu/nửa sau intensity, tỷ lệ thay đổi, 1-2 câu>",
            "doi_hinh":            "<compact trend + hull area avg + width delta, 1-2 câu — trung lập>",
            "hang_thu":            "<def_line_height avg, block style, trend — TUYỆT ĐỐI TRUNG LẬP>",
            "xay_dung":            "<progressive_pass_pct + high_risk_count/total + avg_distance_to_goal, 1-2 câu>",
            "diem_manh":           ["<2-3 điểm mạnh có con số>"],
            "diem_yeu":            ["<2-3 điểm yếu có con số>"],
            "khuyen_nghi_hlv":     ["<2 khuyến nghị hành động cụ thể>"],
        },
    },
    "danh_gia_cau_thu": {
        "doi_1": {
            "pressing_tot":   {"track_id": "<int — track_id pressing_score cao nhất>", "ly_do": "<1 câu>"},
            "width_tot":      {"track_id": "<int — track_id width_contrib_score cao nhất>", "ly_do": "<1 câu>"},
            "can_cai_thien":  {"track_id": "<int — poor_positioning[0].track_id>", "khuyen_nghi": "<1 câu>"},
        },
        "doi_2": {
            "pressing_tot":   {"track_id": "<int>", "ly_do": "<str>"},
            "width_tot":      {"track_id": "<int>", "ly_do": "<str>"},
            "can_cai_thien":  {"track_id": "<int>", "khuyen_nghi": "<str>"},
        },
    },
    "so_sanh_doi_dau": {
        "pressing":             "<PPDA hoặc proximity; so sánh nửa đầu/nửa sau video, cả 2 đội>",
        "doi_hinh":             "<compact phase tấn công/phòng ngự (m²) cả 2 đội — trung lập>",
        "kiem_soat_bong":       "<possession % cả 2 đội>",
        "chien_luoc_phong_ngu": "<def_line avg cả 2 đội, ý đồ chiến thuật — TRUNG LẬP>",
        "su_dung_bien":         "<width avg + width_delta khi có/không bóng cả 2 đội — TRUNG LẬP>",
        "tranh_chap":           "<recoveries tổng số + % opp_half cả 2 đội>",
        "mat_bong":             "<turnovers: total + high_risk_count + high_risk_rate_pct + avg_distance_to_goal cả 2 đội — không phán xét, dùng Contextual Risk>",
        "kien_tao":             "<progressive_pass_pct cả 2 đội, hoặc 'Không đủ dữ liệu.' nếu thiếu>",
    },
    "ket_luan": "<2-3 câu tổng kết chiến thuật, không lặp số liệu đã nêu>",
}

# ── required keys for validation ────────────────────────────────────────────

_REQUIRED_TOP    = {"tong_quan_tran_dau", "danh_gia_doi", "danh_gia_cau_thu",
                    "so_sanh_doi_dau",    "ket_luan"}
_REQUIRED_TEAM   = {
    "chien_thuat", "pressing", "doi_hinh", "hang_thu",
    "xay_dung", "diem_manh", "diem_yeu", "khuyen_nghi_hlv",
}
_REQUIRED_PLAYER = {"track_id"}
_VALID_GRADES: set[str] = set()  # grades removed


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
Bạn là chuyên gia phân tích chiến thuật bóng đá, viết báo cáo ngắn gọn
cho ban huấn luyện sau trận đấu. Phong cách: trực tiếp, có số liệu cụ thể,
mỗi nhận xét gắn với hành động có thể thực hiện.

════════════════════ QUY TẮC BẮT BUỘC ════════════════════

RULE 1 — LUÔN CÓ SỐ LIỆU
Mỗi nhận xét phải trích dẫn ít nhất một con số từ JSON.
✗ "Đội 1 pressing tốt."
✓ "Đội 1 pressing giảm 41% từ nửa đầu (0.12) → nửa sau (0.07)."

RULE 2 — SO SÁNH NỘI BỘ
Chỉ so sánh 2 đội với nhau trong trận này. Không dùng chuẩn ngoài.

RULE 3 — TÊN ĐỘI VÀ CẦU THỦ
Nếu có "team_names" trong JSON, dùng đúng tên đó thay vì "Đội 1"/"Đội 2".
Cầu thủ: gọi là "Cầu thủ #<track_id>". Không đặt tên cầu thủ.

RULE 4 — PHONG CÁCH
Ngắn gọn, trực tiếp. Không dùng "đáng chú ý", "điều này cho thấy",
"cần lưu ý". Mỗi câu nên mang thông tin hành động.

RULE 5 — OUTPUT THUẦN JSON
Trả về JSON theo đúng schema. Không có markdown, không giải thích ngoài.

RULE 6 — SỐ LIỆU TRONG ĐIỂM MẠNH/YẾU
Mỗi phần tử trong diem_manh và diem_yeu bắt buộc có ít nhất 1 con số.

RULE 7 — TRUNG LẬP VỀ HÀNG THỦ VÀ ĐỘ RỘNG
Không phán xét high_block/low_block hay wide/narrow là tốt/xấu.
✗ "Hàng thủ thấp cho thấy đội yếu."
✓ "Đội 2 hàng thủ trung bình 19.7m (low block), nhường thế trận, chờ phản công."

RULE 8 — KHÔNG CÓ ĐIỂM TỔNG / XẾP HẠNG CHỮ CÁI
Tuyệt đối không dùng overall_score hay xếp hạng A/B/C/D/F.
Chỉ mô tả chiến thuật và số liệu thực tế.

RULE 9 — TURNOVERS: DIỄN GIẢI CONTEXTUAL RISK
Với turnovers_final_third: KHÔNG dùng "tệ" hay "yếu kém".
Sử dụng High-Risk/Low-Risk và avg_distance_to_goal để diễn giải chiến thuật.
✗ "Đội 2 mất bóng nhiều ở 1/3 cuối, rất nguy hiểm."
✓ "Đội 2 chấp nhận 4 lần mất bóng ở 1/3 cuối (2 High-Risk, cách khung thành
   TB 14.5 m), thể hiện chủ trương xây dựng tấn công từ sâu chấp nhận rủi ro."

RULE 10 — THIẾU DỮ LIỆU CHUYỀN BÓNG
Nếu total_passes = null: ghi "Không đủ dữ liệu chuyền bóng." — không bịa số.

RULE 11 — KHUYẾN NGHỊ HLV
khuyen_nghi_hlv: mỗi khuyến nghị phải cụ thể, có thể tập luyện ngay.
✗ "Cần cải thiện pressing."
✓ "Bổ sung bài tập pressing trap ở khu vực giữa sân để duy trì cường độ nửa sau video."

RULE 12 — KHÔNG DÙNG "HIỆP"
Video có thể là clip ngắn, không phải trận 90 phút. TUYỆT ĐỐI không dùng
"hiệp một", "hiệp hai", "hiệp 1", "hiệp 2" trong diem_manh, diem_yeu,
nhan_xet_chung, pressing, hay bất kỳ mục nào.
✗ "Cường độ pressing giảm trong hiệp hai."
✓ "PPDA tăng từ 8.2 (nửa đầu video) → 12.1 (nửa sau) — pressing giảm cường độ."

RULE 13 — COMPACT THEO PHASE
Khi có compact_attacking_m2 / compact_defending_m2 trong cau_truc_doi_hinh,
mô tả đội hình theo phase tấn công vs phòng ngự (m² hull area), không dùng
điểm 0-100. Ưu tiên PPDA (press_and_recovery.ppda) cho pressing nếu có.

════════════════════ OUTPUT SCHEMA ════════════════════

{schema_str}

Lưu ý quan trọng:
- "doi_noi_bat" là số nguyên 1, 2, hoặc null (KHÔNG phải chuỗi).
- "diem_manh" và "diem_yeu": mỗi mảng tối thiểu 2 phần tử, mỗi phần tử có số.
- "khuyen_nghi_hlv": mảng 2 phần tử, mỗi phần tử là 1 khuyến nghị hành động.
- track_id trong danh_gia_cau_thu phải lấy từ top_pressers / top_width_users /
  poor_positioning trong phần cau_thu_then_chot của JSON.
- hang_thu: TUYỆT ĐỐI trung lập.
- mat_bong/xay_dung: KHÔNG dùng "tệ" hay "yếu kém".
- so_sanh_doi_dau.kien_tao: nếu thiếu dữ liệu → "Không đủ dữ liệu."
"""

    def _user_prompt(self, match_report: dict) -> str:
        report_json = json.dumps(match_report, ensure_ascii=False, indent=2)

        # Build team name context from the optional "team_names" key
        team_names_block = ""
        tn = match_report.get("team_names", {})
        if tn:
            lines = []
            for k, v in tn.items():
                lines.append(f'  - {k}: "{v}"')
            team_names_block = (
                "\n\nTên đội đã xác định từ bảng tỉ số trong video:\n"
                + "\n".join(lines)
                + "\nHãy dùng đúng tên này (thay cho 'Đội 1'/'Đội 2') trong toàn bộ báo cáo."
            )

        return f"""\
Dưới đây là báo cáo thống kê trận đấu được trích xuất từ video phân tích:{team_names_block}

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

        # Player section — coerce track_id to int
        for doi_key in ("doi_1", "doi_2"):
            cau_thu = data.get("danh_gia_cau_thu", {}).get(doi_key, {})
            for role_key in ("pressing_tot", "width_tot", "can_cai_thien"):
                entry = cau_thu.get(role_key, {})
                if _REQUIRED_PLAYER - set(entry.keys()):
                    raise KeyError(
                        f"danh_gia_cau_thu.{doi_key}.{role_key} missing 'track_id'"
                    )
                try:
                    entry["track_id"] = int(entry["track_id"])
                except (TypeError, ValueError):
                    entry["track_id"] = 0

        return data
