# CODEBASE.md — Agent Charades (Casper Studios Build Day Manila 2026)

## What This Project Does
An AI agent that watches a live camera feed of someone playing charades, interprets their gestures using a vision LLM, and submits guesses via API. Scored on accuracy and speed across 10 rounds.

## Environment
- Language: Python 3.10+
- Package manager: uv
- Runtime: pydantic-ai + OpenRouter (Gemini 2.5 Flash)
- How to run: `uv run -m agent --practice` (local) or `uv run -m agent --live` (event day)
- Config: `.env` file with `LLM_API_KEY` (OpenRouter key)

## Constraints
- **Only `agent/src/agent/prompt.py` can be modified** for the competition
- `core/src/core/practice.py` was modified for local testing only (camera resolution fix)
- 10 rounds × 2 min each, max 10 guesses per round
- Judge uses **semantic matching** (plural/conjugation flexibility)
- Words can be in **English or Filipino/Tagalog**
- Difficulty: Easy (1x), Medium (2x), Hard (3x)

## File Map
| File | What it does |
|------|-------------|
| `agent/src/agent/prompt.py` | ✏️ OUR FILE — system prompt + analyze() logic |
| `agent/src/agent/__main__.py` | 🔒 CLI entry point (practice/live modes) |
| `core/src/core/frame.py` | 🔒 Frame dataclass (PIL Image + timestamp) |
| `core/src/core/practice.py` | 🔒 Local camera capture via ffmpeg (we patched -s 1280x720) |
| `core/src/core/stream.py` | 🔒 LiveKit stream receiver |
| `api/src/api/client.py` | 🔒 HTTP client for game server |
| `.env` | OpenRouter API key + server config |

## Roadmap
- [x] Experiment 1: Base implementation with pydantic-ai + OpenRouter
- [x] Experiment 2: Cross-language support (English + Filipino)
- [x] Fix: Remove multi-guess pipeline (wasted guess budget)
- [ ] Experiment 3: Optimize model accuracy and speed
- [ ] Experiment 4: Tune for abstract/hard difficulty words

## Experiment Log

### Experiment 1 — Base Implementation
**What was built:** Full `prompt.py` rewrite using pydantic-ai `Agent` + `OpenRouterModel` (Gemini 2.5 Flash). System prompt for charades interpretation. 8-frame rolling history for temporal reasoning. Guess deduplication. Adaptive urgency (forces guesses after consecutive skips).
**Result:** ✅ Works. Identifies common gestures (clapping, sleeping, waving, talking, counting, thumbs up). Fairly accurate on concrete actions.
**Files changed:** `prompt.py`, `practice.py` (camera fix)

### Experiment 2 — Cross-Language + Multi-Guess
**What was built:** Added Filipino/Tagalog examples to system prompt. Added multi-guess pipeline (3 guesses per frame via pipe-separated output).
**Result:** ⚠️ Multi-guess worked technically but was wasteful — burned 3 guesses per frame, using 30% of the budget on a single observation. Removed in favor of single-guess.
**Files changed:** `prompt.py`

### Fix — Single Guess Per Frame
**What was built:** Reverted to single-guess output. Model now commits to ONE best answer per frame. Removed `_pending_guesses` queue and pipe-parsing logic.
**Result:** ✅ Much more budget-efficient. Ready for optimization.
**Files changed:** `prompt.py`

## Mistakes & Fixes
| What went wrong | Why it happened | How it was fixed |
|-----------------|----------------|------------------|
| Camera capture failed on MacBook | ffmpeg raw output was non-standard resolution (7226112 bytes) | Added `-s 1280x720` to ffmpeg command in `practice.py` |
| Module crash at import time | `OpenRouterProvider` initialized before `.env` loaded | Lazy-init with `_get_agent()` function |
| Multi-guess wasted budget | 3 guesses per frame × 10 frames = blow through 10-guess limit in 3-4 frames | Removed multi-guess, now single-guess per frame |

## Key Decisions
| Decision | Why |
|----------|-----|
| Gemini 2.5 Flash via OpenRouter | Fastest strong vision model — speed is critical for scoring |
| pydantic-ai (not raw httpx) | Already a dependency, native OpenRouter + BinaryContent support |
| 8-frame rolling history | Charades gestures unfold over time; temporal context improves accuracy |
| Single guess per frame | 10-guess budget is precious — one confident guess > 3 hedged ones |
| JPEG quality=85 | Balance between image quality and payload size for fast LLM calls |

## RESUME HERE
Last completed: Fix — Single Guess Per Frame
Next: **Experiment 3 — Optimize model accuracy and speed**
Things to consider:
- Current prompt is good at concrete actions but untested on abstract/hard words
- Consider: should we observe first (SKIP early frames) and guess later with more confidence, or guess immediately on every frame?
- Filipino word coverage could be expanded in the system prompt
- Image resolution/quality tradeoffs for faster LLM response
- Could try a different model (e.g., `google/gemini-2.5-pro`) for harder rounds if latency is acceptable
- The frame history is 8 frames — experiment with more or fewer
- Consider adding a "describe what you see first, then guess" chain-of-thought approach
