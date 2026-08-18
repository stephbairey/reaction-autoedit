"""Scene boundaries on the movie region — see video_signals.py (computed in the same decode pass as
face motion). ``Scenes`` is re-exported here for stage symmetry."""

from .video_signals import Scenes, detect_cuts  # noqa: F401
