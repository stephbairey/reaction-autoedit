"""Scene boundaries on the movie region (PySceneDetect ContentDetector over the movie_inner crop). [M3]
Output analysis/scenes.json: {"boundaries": [t, ...]}  — used for clean cut points and mid-roll placement.
"""


def detect_scenes(*args, **kwargs):  # pragma: no cover - M3
    raise NotImplementedError("Stage 2 scene detection lands in Milestone 3")
