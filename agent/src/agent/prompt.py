"""System prompt and analysis logic for the guessing game agent.

=== EDIT THIS FILE ===

This is where you define your agent's strategy:
- What system prompt to use
- How to analyze each frame
- When to submit a guess vs. gather more context
"""

from __future__ import annotations

import io
import os
from collections import deque

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from core import Frame

# ---------------------------------------------------------------------------
# Model setup — lazy-initialized after dotenv loads in __main__.py
# ---------------------------------------------------------------------------

_agent: Agent | None = None


def _get_agent() -> Agent:
    """Lazy-init the agent so env vars are available (dotenv loads in __main__)."""
    global _agent
    if _agent is None:
        api_key = (
            os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("LLM_API_KEY")
            or ""
        )
        model = OpenRouterModel(
            "google/gemini-2.5-flash",
            provider=OpenRouterProvider(api_key=api_key),
        )
        _agent = Agent(model, system_prompt=SYSTEM_PROMPT)
    return _agent

# ---------------------------------------------------------------------------
# System prompt — engineered for charades / action guessing
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the world's best charades interpreter. You watch a live camera feed of \
a person acting out a word or phrase using ONLY body language, gestures, and \
mime — no speaking, no props with text.

YOUR TASK: Figure out what word or phrase the person is acting out.

## How to analyze each frame

1. **Observe the body**: What are the hands doing? Arms? Head? Full body posture?
2. **Identify the gesture type**:
   - Mime of an ACTION (swimming, eating, driving, flying, typing, sleeping…)
   - Mime of an OBJECT (holding shape, outlining with hands…)
   - Mime of an ANIMAL (crawling, flapping, slithering…)
   - Mime of a CONCEPT/EMOTION (love = heart shape, cold = shivering, happy = smile gesture…)
   - Acting out a SCENE (movie, book title, song…)
   - Common charades CONVENTIONS: pointing to ear = "sounds like", holding up \
fingers = number of words/syllables
3. **Consider temporal context**: You will be told what was observed in \
previous frames. A gesture may unfold over several seconds — connect the dots.
4. **Generate your best guess**: 1-5 words. Be specific. Prefer the exact \
word/phrase being acted out.

## Rules
- Respond with ONLY your guess (1-5 words), nothing else.
- If you truly cannot see any meaningful gesture (e.g., the person hasn't \
started, or the frame is blurry/black), respond with exactly "SKIP".
- NEVER repeat a guess that was already tried (you'll be told which ones).
- Be AGGRESSIVE with guessing — a wrong guess costs little, but speed matters.
- Think about common charades words: everyday actions, animals, emotions, \
movies, sports, professions, food items.
- Consider abstract or difficult words too: freedom, gravity, time, silence, \
awkward, irony.
"""

# ---------------------------------------------------------------------------
# State — persists across frames within a round
# ---------------------------------------------------------------------------

# Rolling observations from recent frames (last N frame descriptions)
_frame_observations: deque[str] = deque(maxlen=8)

# All guesses we've submitted this round (to avoid repeats)
_previous_guesses: list[str] = []

# Frame counter for adaptive behavior
_frame_count: int = 0

# Consecutive skips tracker
_consecutive_skips: int = 0


def reset_round() -> None:
    """Reset state for a new round."""
    global _frame_count, _consecutive_skips
    _frame_observations.clear()
    _previous_guesses.clear()
    _frame_count = 0
    _consecutive_skips = 0





# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------


async def analyze(frame: Frame) -> str | None:
    """Analyze a single frame and return a guess, or None to skip.

    Uses temporal context (previous frame observations) and guess history
    to make the smartest possible guess. Progressively more aggressive
    as frames accumulate without a correct answer.

    Args:
        frame: A Frame with .image (PIL Image) and .timestamp.

    Returns:
        A text guess string, or None to skip this frame.
    """
    global _frame_count, _consecutive_skips
    _frame_count += 1

    # Convert PIL Image to JPEG bytes for the vision model
    buf = io.BytesIO()
    frame.image.save(buf, format="JPEG", quality=85)
    image_bytes = buf.getvalue()

    # Build the user message with context
    context_parts: list[str] = []

    # Add temporal context from previous frames
    if _frame_observations:
        history = "\n".join(
            f"  Frame {i+1}: {obs}"
            for i, obs in enumerate(_frame_observations)
        )
        context_parts.append(f"PREVIOUS OBSERVATIONS:\n{history}")

    # Add guess deduplication context
    if _previous_guesses:
        tried = ", ".join(f'"{g}"' for g in _previous_guesses)
        context_parts.append(
            f"ALREADY GUESSED (do NOT repeat these): {tried}"
        )

    # Adaptive urgency
    if _consecutive_skips >= 3:
        context_parts.append(
            "⚠️ You have skipped several frames. You MUST make a guess now, "
            "even if uncertain. Any reasonable guess is better than skipping."
        )
    elif _frame_count >= 5 and not _previous_guesses:
        context_parts.append(
            "You've seen multiple frames without guessing. Time to commit "
            "to your best guess — speed matters!"
        )

    # Combine context + instruction
    context_block = "\n\n".join(context_parts) if context_parts else ""

    user_message = (
        f"{context_block}\n\n"
        "Look at this frame from the live camera. "
        "What word or phrase is the person acting out in charades? "
        "Reply with ONLY your guess (1-5 words) or SKIP."
    ).strip()

    # Call the vision model
    try:
        result = await _get_agent().run(
            [
                user_message,
                BinaryContent(data=image_bytes, media_type="image/jpeg"),
            ],
        )
        answer = result.output.strip().strip('"').strip("'")
    except Exception as e:
        print(f"  [agent] LLM error: {e}")
        return None

    # Handle SKIP
    if answer.upper() == "SKIP":
        _consecutive_skips += 1
        # Store a minimal observation even on skip
        _frame_observations.append("(no clear gesture detected)")
        return None

    # We got a guess — reset skip counter
    _consecutive_skips = 0

    # Store this frame's observation for future context
    _frame_observations.append(f"Guessed: {answer}")

    # Deduplicate: don't submit if we already guessed this
    if answer.lower() in [g.lower() for g in _previous_guesses]:
        print(f"  [agent] Duplicate guess avoided: {answer}")
        return None

    # Record the guess
    _previous_guesses.append(answer)

    return answer
