#!/usr/bin/env python
"""
CLI for running benchmark evaluation on transcript files.

Supports single-session and multi-session transcripts (from run_multi_session).

Usage:
  # Evaluate a single transcript (turn-level + dialog-level):
  python -m benchmark.evaluation.run --transcript benchmark/data/transcripts/xxx.json

  # Evaluate multi-session transcript (each session scored, then averaged):
  python -m benchmark.evaluation.run --transcript multi_calc1_beginner_00_xxx.json

  # Dialog-only (faster, skip per-turn scoring):
  python -m benchmark.evaluation.run --transcript xxx.json --dialog-only

  # Evaluate all transcripts in a directory:
  python -m benchmark.evaluation.run --transcript-dir benchmark/data/transcripts

  # Output to file:
  python -m benchmark.evaluation.run --transcript xxx.json --output results.json
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from benchmark.evaluation.evaluator import evaluate_transcript

logger = logging.getLogger("benchmark.evaluation")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_EVAL_DIR = PROJECT_ROOT / "benchmark" / "data" / "evaluations"


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )


async def _run_single(transcript_path: Path, skip_turns: bool, temperature: float) -> dict:
    """Evaluate a single transcript."""
    return await evaluate_transcript(
        transcript_path=transcript_path,
        skip_turns=skip_turns,
        temperature=temperature,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate tutor transcripts from benchmark conversations"
    )
    parser.add_argument(
        "--transcript",
        help="Path to a single transcript JSON file",
    )
    parser.add_argument(
        "--transcript-dir",
        help="Path to directory of transcript JSON files (evaluates all)",
    )
    parser.add_argument(
        "--dialog-only",
        action="store_true",
        help="Skip turn-level evaluation (faster, dialog-level only)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output path for results JSON (default: benchmark/data/evaluations/<stem>_eval_<timestamp>.json)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature for evaluation (default: 0.2)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    _setup_logging(args.verbose)

    if args.transcript and args.transcript_dir:
        parser.error("Use either --transcript or --transcript-dir, not both")
    if not args.transcript and not args.transcript_dir:
        parser.error("Provide --transcript or --transcript-dir")

    paths: list[Path] = []
    if args.transcript:
        p = Path(args.transcript)
        if not p.exists():
            # Try relative to project root (handles /benchmark/... typo)
            alt = PROJECT_ROOT / args.transcript.lstrip("/")
            if alt.exists():
                p = alt
            else:
                parser.error(f"Transcript not found: {p}")
        paths = [p]
    else:
        d = Path(args.transcript_dir)
        if not d.is_dir():
            alt = PROJECT_ROOT / args.transcript_dir.lstrip("/")
            if alt.is_dir():
                d = alt
            else:
                parser.error(f"Directory not found: {d}")
        paths = sorted(d.glob("*.json"))

    if not paths:
        parser.error("No transcript files found")

    results = []
    for path in paths:
        logger.info("Evaluating %s", path)
        try:
            result = await _run_single(
                transcript_path=path,
                skip_turns=args.dialog_only,
                temperature=args.temperature,
            )
            results.append(result)

            # Print summary
            print(f"\n--- {path.name} ---")
            if "error" in result:
                print(f"Error: {result['error']}")
            elif "sessions" in result:
                print(f"Profile: {result.get('profile_id', '?')} ({result.get('num_sessions', 0)} sessions)")
                for i, s in enumerate(result.get("sessions", []), 1):
                    print(f"  Session {i} ({s.get('entry_id', '?')}): turns={s.get('actual_turns', 0)}, "
                          f"combined={s.get('combined_overall_score', 0):.2f}")
                print(f"Turns: {result.get('actual_turns', 0)}")
                print(f"Combined overall (avg): {result.get('combined_overall_score', 0):.2f}")
                print(f"Combined personalization (avg): {result.get('combined_personalization_score', 0):.2f}")
            else:
                print(f"Entry: {result.get('entry_id', '?')}")
                print(f"Turns: {result.get('actual_turns', 0)}")
                print(f"Dialog overall: {result.get('overall_dialog_score', 0):.2f}")
                print(f"Dialog personalization: {result.get('personalization_dialog_score', 0):.2f}")
                if result.get("turn_scores"):
                    print(f"Turn avg overall: {result.get('turn_avg_overall', 0):.2f}")
                    print(f"Turn avg personalization: {result.get('turn_avg_personalization', 0):.2f}")
                print(f"Combined overall (turn 40% + dialog 60%): {result.get('combined_overall_score', 0):.2f}")
                print(f"Combined personalization: {result.get('combined_personalization_score', 0):.2f}")
                if result.get("summary"):
                    print(f"Summary: {result['summary'][:200]}...")
        except Exception as e:
            logger.exception("Failed to evaluate %s", path)
            results.append({"transcript_path": str(path), "error": str(e)})

    # Single file: result is the first (and only) item
    output_data = results[0] if len(results) == 1 else {"evaluations": results}

    # Always save to file; default path when --output not specified
    if args.output:
        out_path = Path(args.output)
    else:
        DEFAULT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if len(paths) == 1:
            stem = paths[0].stem
            out_path = DEFAULT_EVAL_DIR / f"{stem}_eval_{timestamp}.json"
        else:
            out_path = DEFAULT_EVAL_DIR / f"eval_{timestamp}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
