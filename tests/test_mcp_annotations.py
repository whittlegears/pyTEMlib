"""Notes on datasets: one event file, soft delete, history, and refused stale edits."""

import asyncio
import json
import threading

import h5py
import pytest

from pyTEMlib.mcp_server import create_server
from pyTEMlib.mcp_server.annotation_store import AnnotationError, AnnotationStore


@pytest.fixture
def store(tmp_path):
    return AnnotationStore(tmp_path / "notes.jsonl")


def test_a_note_is_created_listed_and_survives_a_restart(store, tmp_path):
    pin = store.create("map_a", "pin", {"x": 1e-6, "y": -2e-6}, "stage_m", "a particle", "particle", ["img_1"], author="alice")
    assert pin["version"] == 1 and pin["author"] == "alice" and pin["coords"] == {"x": 1e-6, "y": -2e-6}
    store.create("map_b", "circle", {"x": 0, "y": 0, "r": 5e-7}, "stage_m")
    assert [n["id"] for n in store.list("map_a")] == [pin["id"]]
    again = AnnotationStore(tmp_path / "notes.jsonl")
    assert again.list("map_a") == store.list("map_a")


@pytest.mark.parametrize("shape, frame, coords, code", [
    ("star", "stage_m", {"x": 0, "y": 0}, "bad_shape"),
    ("pin", "lab", {"x": 0, "y": 0}, "bad_frame"),
    ("range", "stage_m", {"start": 1, "end": 2}, "bad_frame"),
    ("pin", "energy_ev", {"x": 0, "y": 0}, "bad_frame"),
    ("pin", "stage_m", {"x": 0}, "bad_coords"),
    ("pin", "stage_m", {"x": 0, "y": 0, "z": 1}, "bad_coords"),
    ("pin", "stage_m", {"x": float("nan"), "y": 0}, "bad_coords"),
    ("pin", "stage_m", {"x": True, "y": 0}, "bad_coords"),
    ("circle", "stage_m", {"x": 0, "y": 0, "r": 0}, "bad_coords"),
    ("box", "pixel", {"x0": 5, "y0": 0, "x1": 1, "y1": 3}, "bad_coords"),
    ("range", "energy_ev", {"start": 900, "end": 800}, "bad_coords"),
])
def test_bad_geometry_is_refused_and_writes_nothing(store, shape, frame, coords, code):
    with pytest.raises(AnnotationError) as refused:
        store.create("map_a", shape, coords, frame)
    assert refused.value.code == code
    assert not store.path.exists()


def test_an_edit_made_against_an_old_version_is_refused(store):
    note = store.create("spec", "range", {"start": 450.0, "end": 470.0}, "energy_ev", "Ti L edge")
    agent = store.edit(note["id"], {"note": "Ti L2,3"}, author="agent-1", expected_version=1)
    assert agent["version"] == 2 and agent["edited_by"] == "agent-1"
    with pytest.raises(AnnotationError) as stale:
        store.edit(note["id"], {"note": "something else"}, author="bob", expected_version=1)
    assert stale.value.code == "conflict"
    assert store.list("spec")[0]["note"] == "Ti L2,3"


def test_delete_is_soft_and_restore_brings_it_back(store):
    note = store.create("img", "box", {"x0": 0, "y0": 0, "x1": 10, "y1": 20}, "pixel")
    deleted = store.set_deleted(note["id"], True, author="bob")
    assert deleted["deleted"] and store.list("img") == []
    assert store.list("img", include_deleted=True)[0]["deleted_by"] == "bob"
    with pytest.raises(AnnotationError, match="restore it"):
        store.edit(note["id"], {"note": "x"})
    assert not store.set_deleted(note["id"], False)["deleted"]
    assert [e["event"] for e in store.history(note["id"])] == ["create", "delete", "restore"]


def test_an_edit_keeps_geometry_rules_and_fields_it_did_not_name(store):
    note = store.create("map_a", "circle", {"x": 0, "y": 0, "r": 1e-7}, "stage_m", "first", "particle")
    with pytest.raises(AnnotationError):
        store.edit(note["id"], {"shape": "pin"})  # a pin has no r
    moved = store.edit(note["id"], {"shape": "pin", "coords": {"x": 1e-7, "y": 2e-7}})
    assert moved["shape"] == "pin" and moved["note"] == "first" and moved["kind"] == "particle"
    with pytest.raises(AnnotationError) as unknown:
        store.edit(note["id"], {"author": "mallory"})
    assert unknown.value.code == "bad_edit"


def test_people_and_agents_editing_at_once_lose_nothing(store):
    note = store.create("map_a", "pin", {"x": 0, "y": 0}, "stage_m")
    results = []

    def change(who):
        for _ in range(25):
            current = store.list("map_a")[0]
            try:
                store.edit(note["id"], {"note": who}, author=who, expected_version=current["version"])
                results.append(who)
            except AnnotationError as exc:
                assert exc.code == "conflict"

    threads = [threading.Thread(target=change, args=(w,)) for w in ("alice", "agent-1")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    history = store.history(note["id"])
    assert len(history) == 1 + len(results)  # every accepted change is recorded, none overwritten unseen
    assert store.list("map_a")[0]["version"] == len(history)


def test_an_unreadable_line_is_counted_not_fatal(tmp_path):
    path = tmp_path / "notes.jsonl"
    good = AnnotationStore(path).create("map_a", "pin", {"x": 0, "y": 0}, "stage_m")
    with open(path, "a", encoding="utf-8") as f:
        f.write("{not json\n")
    reread = AnnotationStore(path)
    assert reread.unreadable_lines == 1 and reread.list("map_a")[0]["id"] == good["id"]


def _call(mcp, name, args):
    result = asyncio.run(mcp.call_tool(name, args))
    return result.structured_content if result.structured_content is not None else json.loads(result.content[0].text)


def test_the_tools_annotate_only_datasets_that_exist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "image.h5"
    with h5py.File(data, "w") as h5:
        h5["x"] = [1, 2, 3]
    mcp, _tools = create_server(local_roots=[tmp_path], include_basic=False, annotations_path=str(tmp_path / "notes.jsonl"))
    missing = _call(mcp, "annotate", {"dataset": str(tmp_path / "nope.h5"), "shape": "pin", "coords": {"x": 1, "y": 2}, "frame": "pixel"})
    assert missing["error"]["code"] == "unknown_dataset"
    made = _call(mcp, "annotate", {"dataset": str(data), "shape": "pin", "coords": {"x": 1, "y": 2}, "frame": "pixel", "note": "dust", "reported_by": "bob"})
    assert made["author"] == "bob" and made["version"] == 1
    bad = _call(mcp, "annotate", {"dataset": str(data), "shape": "pin", "coords": {"x": 1}, "frame": "pixel"})
    assert bad["error"]["code"] == "bad_coords"
    edited = _call(mcp, "edit_annotation", {"id": made["id"], "note": "a particle", "expected_version": 1})
    assert edited["note"] == "a particle" and edited["version"] == 2
    stale = _call(mcp, "edit_annotation", {"id": made["id"], "note": "x", "expected_version": 1})
    assert stale["error"]["code"] == "conflict"
    _call(mcp, "delete_annotation", {"id": made["id"]})
    assert _call(mcp, "list_annotations", {"dataset": str(data)})["total"] == 0
    _call(mcp, "restore_annotation", {"id": made["id"]})
    listed = _call(mcp, "list_annotations", {"dataset": str(data)})
    assert listed["total"] == 1 and listed["notes"][0]["note"] == "a particle"
    assert [e["event"] for e in _call(mcp, "annotation_history", {"id": made["id"]})["events"]] == ["create", "edit", "delete", "restore"]


def test_without_an_event_file_the_note_tools_are_not_offered(tmp_path):
    mcp, _tools = create_server(local_roots=[tmp_path], include_basic=False)
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert "annotate" not in names and "add_poi" in names


def test_a_write_cut_short_by_a_crash_does_not_swallow_the_next_note(tmp_path):
    path = tmp_path / "notes.jsonl"
    first = AnnotationStore(path).create("map_a", "pin", {"x": 0, "y": 0}, "stage_m")
    with open(path, "ab") as f:
        f.write(b'{"event": "create", "note_id": "torn')  # the process died mid-line
    after = AnnotationStore(path).create("map_a", "pin", {"x": 1, "y": 1}, "stage_m")
    reread = AnnotationStore(path)
    assert {n["id"] for n in reread.list("map_a")} == {first["id"], after["id"]}
    assert reread.unreadable_lines == 1


def test_a_note_whose_create_line_was_lost_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "notes.jsonl"
    store = AnnotationStore(path)
    kept = store.create("map_a", "pin", {"x": 0, "y": 0}, "stage_m")
    lost = store.create("map_a", "pin", {"x": 1, "y": 1}, "stage_m")
    store.edit(lost["id"], {"note": "edited"})
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if not (lost["id"] in l and '"create"' in l)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    reread = AnnotationStore(path)
    assert [n["id"] for n in reread.list("map_a")] == [kept["id"]]
    with pytest.raises(AnnotationError) as unknown:
        reread.edit(lost["id"], {"note": "x"})
    assert unknown.value.code == "unknown_note"


def test_two_processes_sharing_the_file_see_each_other_and_keep_the_version_check(tmp_path):
    path = tmp_path / "notes.jsonl"
    gui, agent = AnnotationStore(path), AnnotationStore(path)
    note = gui.create("map_a", "pin", {"x": 0, "y": 0}, "stage_m", "first")
    assert [n["id"] for n in agent.list("map_a")] == [note["id"]]
    agent.edit(note["id"], {"note": "agent's"}, expected_version=1)
    with pytest.raises(AnnotationError) as stale:
        gui.edit(note["id"], {"note": "gui's"}, expected_version=1)
    assert stale.value.code == "conflict"
    assert gui.list("map_a")[0]["note"] == "agent's"


def test_one_local_file_has_one_set_of_notes_whatever_the_path_spelling(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "image.h5"
    with h5py.File(data, "w") as h5:
        h5["x"] = [1]
    mcp, _tools = create_server(local_roots=[tmp_path], include_basic=False, annotations_path=str(tmp_path / "notes.jsonl"))
    _call(mcp, "annotate", {"dataset": "./image.h5", "shape": "pin", "coords": {"x": 1, "y": 2}, "frame": "pixel"})
    assert _call(mcp, "list_annotations", {"dataset": str(data)})["total"] == 1
    assert _call(mcp, "list_annotations", {"dataset": "image.h5"})["total"] == 1


def test_the_command_line_server_builds_with_notes_in_a_fresh_process(tmp_path):
    import subprocess
    import sys

    code = (
        "from pyTEMlib.mcp_server import create_server; import asyncio; "
        f"mcp, _ = create_server(local_roots=[], include_basic=False, annotations_path={str(tmp_path / 'n.jsonl')!r}); "
        "print(sorted(t.name for t in asyncio.run(mcp.list_tools()) if 'annotation' in t.name or t.name == 'annotate'))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "annotate" in out.stdout and "list_annotations" in out.stdout


def test_a_note_keeps_the_sample_load_its_dataset_recorded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "image.h5"
    with h5py.File(data, "w") as h5:
        h5["x"] = [1, 2, 3]
        h5.attrs.update({"sample_id": "grid 12", "sample_load_id": "load-1", "id": "d1"})
    bare = tmp_path / "bare.h5"
    with h5py.File(bare, "w") as h5:
        h5["x"] = [1]
    mcp, _tools = create_server(local_roots=[tmp_path], include_basic=False, annotations_path=str(tmp_path / "notes.jsonl"))
    made = _call(mcp, "annotate", {"dataset": str(data), "shape": "pin", "coords": {"x": 1, "y": 2}, "frame": "pixel"})
    assert made["sample_id"] == "grid 12" and made["sample_load_id"] == "load-1"
    edited = _call(mcp, "edit_annotation", {"id": made["id"], "note": "moved on"})
    assert edited["sample_load_id"] == "load-1"
    assert "sample_load_id" not in _call(mcp, "annotate", {"dataset": str(bare), "shape": "pin", "coords": {"x": 0, "y": 0}, "frame": "pixel"})
