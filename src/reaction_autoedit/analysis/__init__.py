"""Stage 2 — analysis of the mixed audio track + sampled video frames.

Every analyser writes ONE JSON artifact into ``work/<name>/analysis/`` and is skipped when that
artifact already exists (cheap re-runs; ``--force`` deletes them). All run on CPU; a CUDA GPU is
used automatically when torch sees one (see ``compute.detect``).

Artifacts (planned):
    transcript.json   words + segments with timestamps (faster-whisper)            → transcribe.py
    speakers.json     transcript segments tagged REACTOR | FILM | UNKNOWN          → speakers.py
    peaks.json        reaction peaks: {t, dur, score, kind: laugh|gasp|shout|face} → peaks.py
    music.json        film-audio music spans: {t0, t1, kind: score|song, conf}     → music.py
    scenes.json       movie-region scene boundaries [t, ...] (PySceneDetect)       → scenes.py
    deadair.json      long silent / low-expression spans (compression candidates)  → deadair.py
"""
