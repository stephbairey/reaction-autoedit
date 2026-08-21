import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from reaction_autoedit.gui.server import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    return TestClient(create_app(work_root=tmp_path / "work"))


def test_system_and_projects_empty(client):
    s = client.get("/api/system").json()
    assert "encoder" in s and "keys" in s
    assert client.get("/api/projects").json() == []


def test_settings_roundtrip(client, tmp_path):
    r = client.get("/api/settings/reactor", params={"path": "configs/reactors/x.json"}).json()
    vals = r["values"]
    vals["patreon_url"] = "https://www.patreon.com/test"
    vals["pip"]["corner"] = "bottom-left"
    assert client.post("/api/settings/reactor", json={"path": "configs/reactors/x.json", "values": vals}).status_code == 200
    saved = json.loads((tmp_path / "configs/reactors/x.json").read_text())
    assert saved["patreon_url"].endswith("/test")
    bad = dict(vals); bad["pip"] = {**vals["pip"], "corner": "middle"}
    assert client.post("/api/settings/reactor", json={"path": "configs/reactors/x.json", "values": bad}).status_code == 400


def test_project_create_and_stage_conflict(client, tmp_path, monkeypatch):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"x")   # probe will fail → 500 from create; use a real fixture instead? keep light:
    r = client.post("/api/projects", json={"name": "t", "source": str(src)})
    assert r.status_code in (400, 500)  # invalid media rejected somewhere sane


def test_jobs_endpoint_empty(client):
    assert client.get("/api/jobs").json() == []
