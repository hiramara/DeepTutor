#!/usr/bin/env python
"""
Benchmark Evaluator - Score tutor responses and full dialogs

Uses LLM-as-judge to evaluate:
- Turn-level: 4 dimensions (50% personalization, 25% effectiveness, 25% knowledge_source_alignment)
- Dialog-level: 5 dimensions (50% personalization, 25% quality, 25% knowledge_source_alignment)

Supports:
- Single-session: JSON with 'transcript' and 'entry' keys
- Multi-session: JSON with 'sessions' array; each session has transcript, entry, entry_id
"""

import json
import logging
from pathlib import Path

from benchmark.data_generation.llm_utils import call_llm_json, load_prompt

logger = logging.getLogger("benchmark.evaluation")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Turn: 50% personalization, 25% effectiveness, 25% knowledge_source_alignment (when source present)
TURN_PERSONALIZATION_KEYS = ["profile_adaptation", "misconception_targeting"]
TURN_EFFECTIVENESS_KEYS = ["response_quality", "engagement"]
TURN_SOURCE_ALIGNMENT_KEY = "knowledge_source_alignment"
TURN_WEIGHTS = (0.5, 0.25, 0.25)  # personalization, effectiveness, source_alignment

# Dialog: 50% personalization, 25% quality, 25% knowledge_source_alignment (when source present)
DIALOG_PERSONALIZATION_KEYS = ["adaptation_consistency", "gap_resolution", "success_criteria_met"]
DIALOG_QUALITY_KEYS = ["session_quality", "student_agency"]
DIALOG_SOURCE_ALIGNMENT_KEY = "knowledge_source_alignment"
DIALOG_WEIGHTS = (0.5, 0.25, 0.25)  # personalization, quality, source_alignment


def _format_profile(profile: dict) -> str:
    """Format student profile for prompt."""
    parts = []
    if profile.get("personality"):
        parts.append(f"Personality: {profile['personality']}")
    if profile.get("education_background"):
        parts.append(f"Background: {profile['education_background']}")
    if profile.get("learning_purpose"):
        parts.append(f"Purpose: {profile['learning_purpose']}")
    ks = profile.get("knowledge_state", {})
    if ks.get("known_well"):
        parts.append(f"Known well: {', '.join(ks['known_well'][:5])}")
    if ks.get("partially_known"):
        parts.append(f"Partially known: {', '.join(ks['partially_known'][:5])}")
    if ks.get("unknown"):
        parts.append(f"Unknown: {', '.join(ks['unknown'][:5])}")
    if profile.get("beliefs"):
        parts.append(f"Beliefs (may be misconceptions): {profile['beliefs']}")
    return "\n".join(parts) if parts else "(no profile)"


def _format_gaps(gaps: list) -> str:
    """Format knowledge gaps for prompt."""
    if not gaps:
        return "(no gaps)"
    lines = []
    for g in gaps:
        lines.append(
            f"- {g.get('target_concept', '?')}: {g.get('description', '')[:200]}... "
            f"Manifests as: {g.get('manifestation', '')[:150]}"
        )
    return "\n".join(lines)


def _format_source_content(source_content: dict[int, str] | None, gap_source_pages: list[int] | None = None) -> str:
    """Format source content for evaluator. If gap_source_pages given, only include those pages."""
    if not source_content:
        return "Not applicable"
    pages = sorted(source_content.keys())
    if gap_source_pages:
        pages = [p for p in pages if p in gap_source_pages]
    lines = []
    for p in pages:
        text = source_content.get(p, "")
        if text:
            lines.append(f"### Page {p}\n{text[:2000]}{'...' if len(text) > 2000 else ''}")
    return "\n\n".join(lines) if lines else "Not applicable"


def _format_task(task: dict) -> str:
    """Format task for prompt."""
    parts = []
    if task.get("title"):
        parts.append(f"Title: {task['title']}")
    if task.get("description"):
        parts.append(f"Description: {task['description']}")
    if task.get("success_criteria"):
        parts.append(f"Success criteria: {task['success_criteria']}")
    if task.get("target_gaps"):
        parts.append(f"Target gaps: {task['target_gaps']}")
    if task.get("expected_gap_exposure"):
        parts.append(f"Expected gap exposure: {task['expected_gap_exposure'][:300]}...")
    return "\n".join(parts) if parts else "(no task)"


def _format_transcript(transcript: list[dict]) -> str:
    """Format transcript for prompt."""
    lines = []
    for i, msg in enumerate(transcript, 1):
        role = msg.get("role", "?")
        content = (msg.get("content", "") or "")[:800]
        if len((msg.get("content") or "")) > 800:
            content += "..."
        lines.append(f"[{i}] {role.upper()}: {content}")
    return "\n\n".join(lines)


def _compute_weighted_avg_three(
    scores: dict,
    keys_a: list,
    keys_b: list,
    key_c: str | None,
    weights: tuple[float, float, float],
) -> float:
    """Compute weighted average: w_a * avg_a + w_b * avg_b + w_c * score_c (when key_c present)."""
    w_a, w_b, w_c = weights
    vals_a = [scores.get(k) for k in keys_a if scores.get(k) is not None]
    vals_b = [scores.get(k) for k in keys_b if scores.get(k) is not None]
    val_c = scores.get(key_c) if key_c else None

    avg_a = sum(vals_a) / len(vals_a) if vals_a else 0.0
    avg_b = sum(vals_b) / len(vals_b) if vals_b else 0.0

    if val_c is not None and key_c:
        return w_a * avg_a + w_b * avg_b + w_c * val_c
    total = w_a + w_b
    if total <= 0:
        return avg_a if vals_a else (avg_b if vals_b else 0.0)
    return (w_a / total) * avg_a + (w_b / total) * avg_b


async def evaluate_turn(
    entry: dict,
    transcript_up_to_turn: list[dict],
    student_message: str,
    tutor_response: str,
    turn_index: int,
    temperature: float = 0.2,
) -> dict:
    """
    Evaluate a single tutor response.

    Returns:
        dict with scores, rationale, personalization_subscore, overall_turn_score
    """
    prompt_cfg = load_prompt("eval_turn")
    profile = entry.get("profile", {})
    gaps = entry.get("gaps", [])
    task = entry.get("task", {})
    source_content = entry.get("source_content")

    gap_source_pages = []
    if source_content and gaps:
        for g in gaps:
            gap_source_pages.extend(g.get("source_pages", []))
    source_summary = _format_source_content(source_content, gap_source_pages or None)

    conv_text = _format_transcript(transcript_up_to_turn)

    user_prompt = prompt_cfg["user_template"].format(
        profile_summary=_format_profile(profile),
        gaps_summary=_format_gaps(gaps),
        task_summary=_format_task(task),
        source_content_summary=source_summary,
        conversation_context=conv_text or "(start of conversation)",
        student_message=student_message,
        tutor_response=tutor_response,
    )

    try:
        result = await call_llm_json(
            user_prompt=user_prompt,
            system_prompt=prompt_cfg["system"],
            temperature=temperature,
            max_tokens=1024,
        )
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Turn %d evaluation failed: %s", turn_index, e)
        return {
            "turn_index": turn_index,
            "scores": {},
            "rationale": f"Evaluation failed: {e}",
            "personalization_subscore": 0.0,
            "overall_turn_score": 0.0,
            "error": str(e),
        }

    scores = result.get("scores", {})
    rationale = result.get("rationale", "")

    personalization_subscore = 0.0
    if TURN_PERSONALIZATION_KEYS:
        p_vals = [scores.get(k) for k in TURN_PERSONALIZATION_KEYS if scores.get(k) is not None]
        personalization_subscore = sum(p_vals) / len(p_vals) if p_vals else 0.0

    source_key = TURN_SOURCE_ALIGNMENT_KEY if (source_content and scores.get(TURN_SOURCE_ALIGNMENT_KEY) is not None) else None
    overall = _compute_weighted_avg_three(
        scores,
        TURN_PERSONALIZATION_KEYS,
        TURN_EFFECTIVENESS_KEYS,
        source_key,
        TURN_WEIGHTS,
    )

    return {
        "turn_index": turn_index,
        "scores": scores,
        "rationale": rationale,
        "personalization_subscore": round(personalization_subscore, 2),
        "overall_turn_score": round(overall, 2),
    }


async def evaluate_dialog(
    entry: dict,
    transcript: list[dict],
    temperature: float = 0.2,
) -> dict:
    """
    Evaluate the entire tutoring dialog.

    Returns:
        dict with scores, summary, personalization_dialog_score, overall_dialog_score
    """
    prompt_cfg = load_prompt("eval_dialog")
    profile = entry.get("profile", {})
    gaps = entry.get("gaps", [])
    task = entry.get("task", {})
    source_content = entry.get("source_content")

    gap_source_pages = []
    if source_content and gaps:
        for g in gaps:
            gap_source_pages.extend(g.get("source_pages", []))
    source_summary = _format_source_content(source_content, gap_source_pages or None)

    user_prompt = prompt_cfg["user_template"].format(
        profile_summary=_format_profile(profile),
        gaps_summary=_format_gaps(gaps),
        task_summary=_format_task(task),
        source_content_summary=source_summary,
        transcript=_format_transcript(transcript),
    )

    try:
        result = await call_llm_json(
            user_prompt=user_prompt,
            system_prompt=prompt_cfg["system"],
            temperature=temperature,
            max_tokens=1024,
        )
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Dialog evaluation failed: %s", e)
        return {
            "scores": {},
            "summary": f"Evaluation failed: {e}",
            "personalization_dialog_score": 0.0,
            "overall_dialog_score": 0.0,
            "error": str(e),
        }

    scores = result.get("scores", {})
    summary = result.get("summary", "")

    personalization_subscore = 0.0
    if DIALOG_PERSONALIZATION_KEYS:
        p_vals = [scores.get(k) for k in DIALOG_PERSONALIZATION_KEYS if scores.get(k) is not None]
        personalization_subscore = sum(p_vals) / len(p_vals) if p_vals else 0.0

    source_key = DIALOG_SOURCE_ALIGNMENT_KEY if (source_content and scores.get(DIALOG_SOURCE_ALIGNMENT_KEY) is not None) else None
    overall = _compute_weighted_avg_three(
        scores,
        DIALOG_PERSONALIZATION_KEYS,
        DIALOG_QUALITY_KEYS,
        source_key,
        DIALOG_WEIGHTS,
    )

    return {
        "scores": scores,
        "summary": summary,
        "personalization_dialog_score": round(personalization_subscore, 2),
        "overall_dialog_score": round(overall, 2),
    }


def _load_entry_by_id(entry_id: str) -> dict | None:
    """Try to load entry from benchmark JSONL by entry_id (for backward compat)."""
    generated_dir = PROJECT_ROOT / "benchmark" / "data" / "generated"
    for jsonl_path in sorted(generated_dir.glob("*.jsonl")):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("entry_id") == entry_id:
                    return entry
    return None


async def _evaluate_single_session(
    transcript: list[dict],
    entry: dict,
    entry_id: str,
    skip_turns: bool,
    temperature: float,
) -> dict:
    """Evaluate one session (single dialog). Returns result dict."""
    result = {
        "entry_id": entry_id,
        "actual_turns": len([m for m in transcript if m.get("role") == "student"]),
        "turn_scores": [],
        "dialog_scores": {},
        "personalization_dialog_score": 0.0,
        "overall_dialog_score": 0.0,
        "summary": "",
    }

    if not skip_turns:
        turn_idx = 0
        for i in range(len(transcript)):
            msg = transcript[i]
            if msg.get("role") == "student" and i + 1 < len(transcript):
                next_msg = transcript[i + 1]
                if next_msg.get("role") == "tutor":
                    student_msg = msg.get("content", "")
                    tutor_msg = next_msg.get("content", "")
                    turn_idx += 1
                    conv_before = transcript[:i]
                    turn_result = await evaluate_turn(
                        entry=entry,
                        transcript_up_to_turn=conv_before,
                        student_message=student_msg,
                        tutor_response=tutor_msg,
                        turn_index=turn_idx,
                        temperature=temperature,
                    )
                    result["turn_scores"].append(turn_result)
                    logger.info(
                        "Turn %d: overall=%.2f, personalization=%.2f",
                        turn_idx,
                        turn_result["overall_turn_score"],
                        turn_result["personalization_subscore"],
                    )

    dialog_result = await evaluate_dialog(
        entry=entry,
        transcript=transcript,
        temperature=temperature,
    )
    result["dialog_scores"] = dialog_result.get("scores", {})
    result["personalization_dialog_score"] = dialog_result.get("personalization_dialog_score", 0.0)
    result["overall_dialog_score"] = dialog_result.get("overall_dialog_score", 0.0)
    result["summary"] = dialog_result.get("summary", "")

    turn_avg_overall = 0.0
    turn_avg_personalization = 0.0
    if result["turn_scores"]:
        turn_avg_overall = sum(t["overall_turn_score"] for t in result["turn_scores"]) / len(
            result["turn_scores"]
        )
        turn_avg_personalization = sum(
            t["personalization_subscore"] for t in result["turn_scores"]
        ) / len(result["turn_scores"])

    result["turn_avg_overall"] = round(turn_avg_overall, 2)
    result["turn_avg_personalization"] = round(turn_avg_personalization, 2)

    if result["turn_scores"]:
        result["combined_overall_score"] = round(
            0.4 * turn_avg_overall + 0.6 * result["overall_dialog_score"], 2
        )
        result["combined_personalization_score"] = round(
            0.4 * turn_avg_personalization + 0.6 * result["personalization_dialog_score"], 2
        )
    else:
        result["combined_overall_score"] = result["overall_dialog_score"]
        result["combined_personalization_score"] = result["personalization_dialog_score"]

    return result


async def evaluate_transcript(
    transcript_path: str | Path,
    skip_turns: bool = False,
    temperature: float = 0.2,
) -> dict:
    """
    Evaluate a transcript file (from conversation runner output).

    Supports single-session (transcript + entry) and multi-session (sessions array).
    For multi-session, evaluates each session and returns aggregated scores.

    Args:
        transcript_path: Path to transcript JSON
        skip_turns: If True, only run dialog-level evaluation (faster)
        temperature: LLM temperature for evaluation

    Returns:
        Full evaluation result
    """
    path = Path(transcript_path)
    if not path.exists():
        raise FileNotFoundError(f"Transcript not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Multi-session format
    if "sessions" in data:
        sessions = data["sessions"]
        profile_id = data.get("profile_id", "unknown")
        session_results = []

        for i, sess in enumerate(sessions):
            transcript = sess.get("transcript", [])
            entry = sess.get("entry")
            entry_id = sess.get("entry_id", f"session_{i+1}")

            if not entry:
                entry = _load_entry_by_id(entry_id)
            if not entry:
                logger.warning(
                    "Session %s has no entry; skipping (run multi-session again to save entry)",
                    entry_id,
                )
                continue

            logger.info("Evaluating session %d/%d: %s", i + 1, len(sessions), entry_id)
            sess_result = await _evaluate_single_session(
                transcript=transcript,
                entry=entry,
                entry_id=entry_id,
                skip_turns=skip_turns,
                temperature=temperature,
            )
            session_results.append(sess_result)

        if not session_results:
            raise ValueError(
                "No sessions could be evaluated (missing 'entry' in each session; "
                "re-run multi-session to save entries)"
            )

        # Aggregate
        n = len(session_results)
        return {
            "profile_id": profile_id,
            "transcript_path": str(path),
            "num_sessions": n,
            "sessions": session_results,
            "actual_turns": sum(r["actual_turns"] for r in session_results),
            "combined_overall_score": round(
                sum(r["combined_overall_score"] for r in session_results) / n, 2
            ),
            "combined_personalization_score": round(
                sum(r["combined_personalization_score"] for r in session_results) / n, 2
            ),
            "overall_dialog_score": round(
                sum(r["overall_dialog_score"] for r in session_results) / n, 2
            ),
            "personalization_dialog_score": round(
                sum(r["personalization_dialog_score"] for r in session_results) / n, 2
            ),
        }

    # Single-session format
    transcript = data.get("transcript", [])
    entry = data.get("entry", {})

    if not entry:
        raise ValueError("Transcript must contain 'entry' (benchmark entry with profile, gaps, task)")

    result = await _evaluate_single_session(
        transcript=transcript,
        entry=entry,
        entry_id=data.get("entry_id", "unknown"),
        skip_turns=skip_turns,
        temperature=temperature,
    )
    result["transcript_path"] = str(path)
    return result
