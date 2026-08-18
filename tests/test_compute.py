from reaction_autoedit import compute


def test_cpu_profile_when_gpu_disabled(monkeypatch):
    compute.detect.cache_clear()
    monkeypatch.setattr(compute, "_torch_cuda", lambda: (False, None))
    monkeypatch.setattr(compute.ffmpeg, "available", lambda: True)
    monkeypatch.setattr(compute.ffmpeg, "encoder_works", lambda name: False)
    p = compute.detect(prefer_gpu=True)
    assert p.device == "cpu"
    assert p.video_encoder == "libx264"
    assert p.whisper_model == "small"
    assert "-crf" in p.encoder_args(preview=True)
    compute.detect.cache_clear()


def test_gpu_profile(monkeypatch):
    compute.detect.cache_clear()
    monkeypatch.setattr(compute, "_torch_cuda", lambda: (True, "FakeGPU"))
    monkeypatch.setattr(compute.ffmpeg, "available", lambda: True)
    monkeypatch.setattr(compute.ffmpeg, "encoder_works", lambda name: name == "h264_nvenc")
    p = compute.detect(prefer_gpu=True)
    assert p.device == "cuda" and p.gpu_name == "FakeGPU"
    assert p.video_encoder == "h264_nvenc"
    assert p.whisper_model == "medium"
    compute.detect.cache_clear()
