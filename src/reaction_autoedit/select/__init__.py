"""Stage 3 — two-budget selection → EDL. [M4]

Inputs: analysis artifacts (transcript, speakers, peaks, music, scenes, deadair), TitleConfig
(runtime target, clip cap, withhold toggle), ReactorConfig.

Algorithm sketch:
  1. Budget A (narrative spine): FILM-tagged transcript → beats. Heuristic scorer (dialogue density,
     named entities, scene starts) or optional LLM pass (Anthropic) that returns
     [{"t0","t1","label","importance"}]. Cover chronologically; each beat rendered as movie-large
     slices ≤ clip_cap separated by reactor-large cutaways.
  2. Budget B (reaction peaks): remaining runtime → highest-scoring peaks, in chronological position,
     as reactor-large segments (movie-large slice of what triggered it when the music tier allows).
  3. Constraints: strictly chronological; song spans excluded or reactor-large only; score spans
     minimised; cut on scene boundaries where within ±1.5 s; layout hysteresis (TitleConfig.layout_min_s).
  4. Withhold-the-climax: drop/trim the 2–3 top peaks in the last act; insert a `cta` segment
     (reactor-large + lower third) in their place.
  5. Runtime loop: while total > target: shrink Budget B, then compress Budget A slices.
Output: EDL (see reaction_autoedit.edl) written to work/<name>/edl.json.
"""


def select(*args, **kwargs):  # pragma: no cover - M4
    raise NotImplementedError("Stage 3 selection lands in Milestone 4")
