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
# System prompt — charades expert with cross-language support
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the world's best charades interpreter. You watch a live camera feed \
of a person acting out a word or phrase using ONLY body language, gestures, \
and mime — no speaking, no props with text.

YOUR TASK: Identify the word or phrase being acted out. The answer could be \
in ENGLISH or FILIPINO (Tagalog). You must consider BOTH languages.

## How to analyze each frame

1. **Observe carefully**: hands, arms, head, full body posture, facial expression, \
movement direction, speed, repetition.
2. **Identify the gesture category**:
   - ACTIONS/VERBS: swimming (langoy), eating (kain), driving, flying (lipad), \
typing, sleeping (tulog), dancing (sayaw), cooking (luto), crying (iyak)…
   - OBJECTS: shaping with hands, outlining, holding invisible items…
   - ANIMALS: crawling, flapping, slithering, pouncing…
   - EMOTIONS/CONCEPTS: love (mahal), cold (malamig), happy (masaya), \
anger (galit), fear (takot), freedom (kalayaan), time (oras)…
   - PROFESSIONS: doctor, teacher (guro), police (pulis), chef…
   - ABSTRACT IDEAS: gravity, silence (katahimikan), irony, awkward, \
beauty (ganda), strength (lakas), peace (kapayapaan)…
   - SCENES/TITLES: movie, song, book, TV show titles…
   - CHARADES CONVENTIONS: ear point = "sounds like", fingers up = word/syllable count
3. **Consider temporal context**: Previous frames are provided. Gestures unfold \
over time — connect the sequence.
4. **Think across difficulty levels**:
   - Easy: obvious everyday actions (waving, eating, sleeping)
   - Medium: nuanced concepts or less obvious actions (surfing, meditating)
   - Hard: abstract ideas, complex phrases, or unusual words (déjà vu, gravity, \
pag-asa, kalayaan, bayanihan)

## Output format
Respond with ONLY your single best guess (1-5 words), nothing else.
The guess can be in English or Filipino — whichever you believe is the answer.

Examples:
- "swimming"
- "mahal"
- "cooking"
- "kalayaan"
- "SKIP"

## Rules
- If you cannot see any meaningful gesture, respond with exactly "SKIP".
- NEVER repeat a guess already tried (you'll be told which ones).
- Be AGGRESSIVE — wrong guesses cost little, speed wins.
- The judge uses semantic matching, so "sharks" matches "shark" and \
"swimming" matches "swim". Focus on the CORE CONCEPT.
- You only get 10 guesses per round, so make each one count.
- Prefer the MOST LIKELY answer. Don't hedge — commit to one guess.
"""

# ---------------------------------------------------------------------------
# State — persists across frames within a round
# ---------------------------------------------------------------------------

# Rolling observations from recent frames
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
    """Analyze a single frame and return a single guess, or None to skip.

    Strategy:
    - Sends frame + temporal context to vision LLM
    - Model returns ONE best guess (preserves the 10-guess budget)
    - Deduplicates against all prior guesses

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

    if _frame_observations:
        history = "\n".join(
            f"  Frame {i+1}: {obs}"
            for i, obs in enumerate(_frame_observations)
        )
        context_parts.append(f"PREVIOUS OBSERVATIONS:\n{history}")

    if _previous_guesses:
        tried = ", ".join(f'"{g}"' for g in _previous_guesses)
        context_parts.append(
            f"ALREADY GUESSED (do NOT repeat): {tried}"
        )

    # Adaptive urgency
    if _consecutive_skips >= 2:
        context_parts.append(
            "⚠️ Multiple frames skipped. You MUST guess now — even if uncertain."
        )
    elif _frame_count >= 4 and not _previous_guesses:
        context_parts.append(
            "Several frames seen with no guess yet. Commit to your best guess!"
        )

    context_block = "\n\n".join(context_parts) if context_parts else ""

    user_message = (
        f"{context_block}\n\n"
        "What word or phrase is this person acting out? "
        "Give your SINGLE best guess (English or Filipino). "
        "Or reply SKIP."
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
        _frame_observations.append("(no clear gesture detected)")
        return None

    _consecutive_skips = 0

    # Store observation for temporal context
    _frame_observations.append(f"Guessed: {answer}")

    # Deduplicate: don't submit if we already guessed this
    if answer.lower() in [g.lower() for g in _previous_guesses]:
        print(f"  [agent] Duplicate guess avoided: {answer}")
        return None

    # Record and submit the guess
    _previous_guesses.append(answer)
    return answer

