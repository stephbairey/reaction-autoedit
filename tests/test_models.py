from reaction_autoedit.models import FaceRegion, FrameInfo, Geometry, Rect


def test_rect_iou_and_even():
    a = Rect(x=0, y=0, w=10, h=10)
    b = Rect(x=5, y=5, w=10, h=10)
    assert abs(a.iou(b) - 25 / 175) < 1e-9
    assert Rect(x=3, y=5, w=7, h=9).even() == Rect(x=2, y=4, w=6, h=8)
    assert Rect(x=3, y=5, w=7, h=9).ffmpeg_crop() == "crop=6:8:2:4"


def test_geometry_clamps_inner():
    g = Geometry(frame=FrameInfo(w=100, h=100, fps=30), movie=Rect(x=10, y=10, w=50, h=50),
                 movie_inner=Rect(x=0, y=0, w=80, h=80), face=FaceRegion(x=70, y=0, w=30, h=30))
    assert g.movie_inner == Rect(x=10, y=10, w=50, h=50)
    assert g.notes
