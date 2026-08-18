"""Runtime compute detection: CPU vs GPU, hardware encoders, sensible defaults.

The tool must run on a CPU-only machine; GPU is an opportunistic speed-up, never a requirement.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from . import ffmpeg


@dataclass
class ComputeProfile:
    device: str = "cpu"                     # "cpu" | "cuda"
    gpu_name: str | None = None
    cpu_count: int = 1
    video_encoder: str = "libx264"          # "libx264" | "h264_nvenc" | ...
    whisper_model: str = "small"
    whisper_compute_type: str = "int8"
    notes: list[str] = field(default_factory=list)

    @property
    def render_jobs(self) -> int:
        """Parallel ffmpeg segment encodes. libx264 already threads; keep this modest."""
        if self.video_encoder != "libx264":
            return 2
        return max(1, min(4, self.cpu_count // 3))

    def encoder_args(self, preview: bool = False) -> list[str]:
        if self.video_encoder == "h264_nvenc":
            if preview:
                return ["-c:v", "h264_nvenc", "-preset", "p1", "-cq", "32"]
            return ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "20", "-b:v", "0"]
        if preview:
            return ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-tune", "fastdecode"]
        return ["-c:v", "libx264", "-preset", "medium", "-crf", "19"]


def _torch_cuda() -> tuple[bool, str | None]:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
    except Exception:  # torch absent or broken; that's fine
        pass
    return False, None


@lru_cache(maxsize=4)
def detect(prefer_gpu: bool = True, encoder: str | None = None) -> ComputeProfile:
    """Detect the compute profile. Cached; pass ``encoder`` to force a specific ffmpeg encoder."""
    p = ComputeProfile(cpu_count=os.cpu_count() or 1)
    forced_device = os.environ.get("RAE_DEVICE")
    if forced_device in ("cpu", "cuda"):
        p.device = forced_device
        p.notes.append(f"device forced via RAE_DEVICE={forced_device}")
    elif prefer_gpu:
        ok, name = _torch_cuda()
        if ok:
            p.device, p.gpu_name = "cuda", name
    if p.device == "cuda":
        p.whisper_model, p.whisper_compute_type = "medium", "float16"
    else:
        p.notes.append("CPU mode: whisper 'small' int8; analysis will be slow but works")

    if encoder:
        p.video_encoder = encoder
        p.notes.append(f"encoder forced: {encoder}")
    elif ffmpeg.available():
        env_enc = os.environ.get("RAE_ENCODER")
        if env_enc:
            p.video_encoder = env_enc
        elif prefer_gpu and ffmpeg.encoder_works("h264_nvenc"):
            p.video_encoder = "h264_nvenc"
        else:
            p.video_encoder = "libx264"
    else:
        p.notes.append("ffmpeg not found; rendering unavailable")
    return p
