"""
LLM-powered tactical report generator.

Converts the structured evaluation dict produced by ``rule_engine.evaluate_tactics``
into a natural-language tactical report.  Supports two backends:

  • **Ollama** (default) — local inference via ``http://localhost:11434``.
    Start the server with ``ollama serve`` and pull the model with
    ``ollama pull llama3`` before calling this module.

  • **OpenAI** — cloud inference via the ``openai`` Python package.
    Requires the ``OPENAI_API_KEY`` environment variable.

Backend selection
-----------------
Priority order (first non-empty value wins):

  1. The ``llm_provider`` argument passed to :func:`generate_report`.
  2. The ``LLM_PROVIDER`` environment variable (``ollama`` | ``openai``).
  3. Falls back to ``"ollama"``.

Optionally override the OpenAI model via ``OPENAI_MODEL`` (default ``gpt-4o-mini``).
"""

from __future__ import annotations

import os
import textwrap
from typing import Any

import requests


# ── prompt constants ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a professional football tactical analyst. "
    "Write concise, insightful tactical reports based on structured match data."
)

_USER_TEMPLATE = textwrap.dedent("""\
    Based on the following tactical analysis of a football match, write a report
    with exactly three labelled sections:

      1. Match Overview
      2. Team-by-Team Analysis
      3. Tactical Conclusion

    Keep the entire report under 400 words. Use clear, professional football
    language — avoid restating raw labels verbatim; interpret them instead.

    --- TACTICAL DATA ---
    {data}
""")

# ── backend configuration ─────────────────────────────────────────────────────

_OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
_OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "llama3")
_OPENAI_MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_REQUEST_TIMEOUT = 120   # seconds


# ── prompt builder ────────────────────────────────────────────────────────────

def _format_evaluation(evaluation: dict[Any, Any]) -> str:
    """
    Render the evaluation dict as indented key-value text for the LLM prompt.

    Raw numbers are intentionally excluded; only labelled assessments are sent
    so the model focuses on tactical interpretation, not re-classification.
    """
    lines: list[str] = []

    for team_id, val in evaluation.items():
        if team_id == "match_events":
            continue
        if not isinstance(val, dict):
            continue

        lines.append(f"Team {team_id}:")
        lines.append(f"  Dominant formation : {val.get('formation', 'N/A')}")
        lines.append(f"  Compactness        : {val.get('compactness_label', 'N/A')}")
        lines.append(f"  Pressing style     : {val.get('pressing_label', 'N/A')}")
        lines.append(f"  Speed trend        : {val.get('speed_trend', 'N/A')}")
        lines.append(f"  Possession style   : {val.get('possession_label', 'N/A')}")
        flags = val.get("flags") or []
        if flags:
            lines.append(f"  Tactical flags     : {', '.join(flags)}")
        lines.append("")

    events: list[str] = evaluation.get("match_events") or []
    if events:
        lines.append("Notable match events:")
        for ev in events:
            lines.append(f"  • {ev}")

    return "\n".join(lines)


# ── backend implementations ───────────────────────────────────────────────────

def _query_ollama(user_text: str) -> str:
    """Send a generate request to a local Ollama instance."""
    payload = {
        "model":  _OLLAMA_MODEL,
        "system": _SYSTEM_PROMPT,
        "prompt": user_text,
        "stream": False,
    }
    try:
        resp = requests.post(_OLLAMA_URL, json=payload, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError(
            f"Cannot reach Ollama at {_OLLAMA_URL}. "
            "Start the server with `ollama serve` and pull the model with "
            f"`ollama pull {_OLLAMA_MODEL}`."
        ) from exc

    data = resp.json()
    report = data.get("response", "").strip()
    if not report:
        raise ValueError(
            f"Ollama returned an empty response.  Full payload: {data}"
        )
    return report


def _query_openai(user_text: str) -> str:
    """Send a chat-completion request to the OpenAI API."""
    try:
        import openai  # optional dependency
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is required for OpenAI backend. "
            "Install it with: pip install openai"
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set."
        )

    client   = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=_OPENAI_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_text},
        ],
        max_tokens=600,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


# ── public API ────────────────────────────────────────────────────────────────

def generate_report(
    evaluation:    dict,
    llm_provider:  str | None = None,
) -> str:
    """
    Generate a natural-language tactical report from a rule-engine evaluation.

    Parameters
    ----------
    evaluation :
        Output of ``rule_engine.evaluate_tactics()``.
    llm_provider :
        ``"ollama"`` or ``"openai"``.  Overrides the ``LLM_PROVIDER`` env var.
        If neither is set, defaults to ``"ollama"``.

    Returns
    -------
    str
        A formatted tactical report of ≤ 400 words with three sections:
        *Match Overview*, *Team-by-Team Analysis*, *Tactical Conclusion*.

    Raises
    ------
    ValueError
        If an unknown provider name is supplied.
    ConnectionError
        If the Ollama server is unreachable.
    EnvironmentError
        If ``OPENAI_API_KEY`` is missing when using the OpenAI backend.
    """
    provider = (
        llm_provider
        or os.getenv("LLM_PROVIDER", "ollama")
    ).lower().strip()

    data_text = _format_evaluation(evaluation)
    user_text = _USER_TEMPLATE.format(data=data_text)

    if provider == "ollama":
        return _query_ollama(user_text)
    elif provider == "openai":
        return _query_openai(user_text)
    else:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            "Valid options: 'ollama', 'openai'."
        )


# ── example output ────────────────────────────────────────────────────────────
# generate_report(evaluation, llm_provider="ollama")
#
# Returns a string similar to:
#
# --- Match Overview ---
# This match featured a tactically disciplined encounter between two sides with
# contrasting approaches to territory and ball retention.  Team 1 maintained
# dominant possession (58%), using a fluid 4-3-3 shape to control the tempo,
# while Team 2 responded with a compact 4-4-2 that absorbed pressure before
# attempting direct transitions.
#
# --- Team-by-Team Analysis ---
# Team 1 demonstrated a mid-block pressing structure during the first half,
# occasionally intensifying to a high press around frames 270–390, likely
# triggered by a turnover opportunity.  Their compactness rating was "compact"
# (≈18 m mean inter-player distance), and their pace remained consistent
# throughout—suggesting excellent physical preparation.
#
# Team 2 showed signs of fatigue in the second half, with their average speed
# dropping more than 15% compared to the opening period.  A tactical shift
# (4-4-2 → 4-5-1 at frame 510) indicates the coaching staff responded by
# adding an extra midfielder to protect space behind the defensive line.
# Their stretched shape (≈26 m compactness) left wide channels exposed.
#
# --- Tactical Conclusion ---
# Team 1's possession-based game plan proved effective at dictating rhythm,
# though they should capitalise on opponent fatigue windows by pushing their
# high press earlier in transitions.  Team 2 must improve their defensive
# compactness to reduce the risk of being overloaded in central areas during
# high-press phases.
