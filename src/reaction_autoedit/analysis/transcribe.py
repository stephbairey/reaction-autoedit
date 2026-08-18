"""Transcription (faster-whisper) over the whole mixed track. [M2]

Interface::

    transcribe(project, model=None, device=None, force=False) -> Path  # analysis/transcript.json

Output schema::

    {"model": "small", "language": "en",
     "segments": [{"id": 0, "start": 12.3, "end": 15.1, "text": "...", "words": [{"w": "..", "s": 12.3, "e": 12.6}]}]}

Notes: model/compute type default from ``compute.detect()`` (small/int8 on CPU, medium/float16 on
CUDA). Audio is extracted to 16 kHz mono WAV first (ffmpeg) and cached in analysis/audio16k.wav.
"""


def transcribe(*args, **kwargs):  # pragma: no cover - M2
    raise NotImplementedError("Stage 2 transcription lands in Milestone 2")
