"""Server and scoring tests for the recognition study. Run: pytest tests"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "recognition"))

import app as server  # noqa: E402
import build_dataset  # noqa: E402
import score  # noqa: E402
from vocabulary import CLASSES, SYNONYMS, canonical, classes_in  # noqa: E402

COLUMNS = ["image_id", "path", "algorithm", "threads_target", "threads_actual",
           "source", "split", "classes"]


@pytest.fixture
def cfg(tmp_path):
    rows = []
    for i in range(12):
        algo = "abcd"[i % 4]
        rel = f"images/{i}.png"
        (tmp_path / "images").mkdir(exist_ok=True)
        (tmp_path / rel).write_bytes(b"\x89PNG" + bytes([i]))
        rows.append({"image_id": f"{i:012x}", "path": rel, "algorithm": algo,
                     "threads_target": 1000 * (i % 3 + 1), "threads_actual": 1000,
                     "source": f"{i}.jpg", "split": "test", "classes": "dog|person"})
    with open(tmp_path / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return {"usernames": ["Ann", "bob", "cy"], "raters_per_image": 2,
            "data_dir": str(tmp_path), "order_seed": 1}


@pytest.fixture
def client(cfg):
    return server.create_app(cfg).test_client()


def login(client, name="ann", agree=True):
    result = client.post("/api/login", json={"username": name})
    if agree and result.status_code == 200:
        client.post("/api/consent", json={"agreed": True, "name": "Ann Example"})
    return result


def answer(client, image_id, labels=("dog",), **extra):
    return client.post("/api/answer", json={
        "id": image_id, "labels": list(labels), "unidentified": not labels,
        "ms": 1200, "w": 640, "h": 480, "dpr": 2, "touch": False, **extra})


def test_assignment_is_equal_and_each_image_has_two_raters():
    rows = [{"image_id": f"{i:012x}", "algorithm": str(i % 5), "threads_target": str(i % 7)}
            for i in range(300)]
    tasks = server.build_assignments(rows, ["a", "b", "c"], 2, 1)
    assert {len(v) for v in tasks.values()} == {200}
    counts = Counter(i for ids in tasks.values() for i in ids)
    assert set(counts.values()) == {2}
    assert tasks["a"] != sorted(tasks["a"])


def test_login_is_case_insensitive_and_rejects_strangers(client):
    assert login(client, " ANN ").status_code == 200
    assert login(client, "eve").status_code == 401


def test_task_requires_login_and_images_are_blinded(client):
    assert client.get("/api/state").status_code == 401
    assert client.get("/img/000000000000").status_code == 401
    login(client)
    state = client.get("/api/state").get_json()
    assert state["total"] == 8 and state["done"] == 0
    image_id = state["queue"][0]["id"]
    image = client.get(f"/img/{image_id}")
    assert b"PNG" in image.data
    assert "Content-Disposition" not in image.headers
    assert client.get("/img/../manifest.csv").status_code == 404


def test_answers_persist_and_resume_in_a_new_session(cfg):
    application = server.create_app(cfg)
    first = application.test_client()
    login(first)
    queue = first.get("/api/state").get_json()["queue"]
    assert answer(first, queue[0]["id"]).get_json()["done"] == 1

    second = server.create_app(cfg).test_client()
    login(second)
    state = second.get("/api/state").get_json()
    assert state["done"] == 1
    assert state["queue"][0]["id"] == queue[1]["id"]


def test_editing_keeps_one_row_and_first_timing(client):
    login(client)
    image_id = client.get("/api/state").get_json()["queue"][0]["id"]
    answer(client, image_id, ["dog"])
    answer(client, image_id, ["cat"], ms=9)
    assert client.get("/api/state").get_json()["done"] == 1
    item = client.get("/api/previous").get_json()["item"]
    assert item["labels"] == ["cat"]


def test_invalid_answers_are_refused(client):
    login(client)
    image_id = client.get("/api/state").get_json()["queue"][0]["id"]
    bad = [
        {"id": image_id, "labels": [], "unidentified": False},
        {"id": image_id, "labels": ["dog"], "unidentified": True},
        {"id": image_id, "labels": "dog"},
        {"id": image_id, "labels": [1]},
        {"id": image_id, "labels": ["x"] * 21},
    ]
    for body in bad:
        assert client.post("/api/answer", json=body).status_code == 400


def test_images_outside_the_users_task_are_refused(cfg):
    application = server.create_app(cfg)
    ann, bob = application.test_client(), application.test_client()
    login(ann)
    login(bob, "bob")
    state = bob.get("/api/state").get_json()
    mine = {i["id"] for i in ann.get("/api/next?after=-1").get_json()["queue"]}
    other = next(i["id"] for i in state["queue"] if i["id"] not in mine)
    assert answer(ann, other).status_code == 403


def test_previous_walks_backwards(client):
    login(client)
    queue = client.get("/api/state").get_json()["queue"]
    answer(client, queue[0]["id"], ["dog"])
    answer(client, queue[1]["id"], [])
    item = client.get(f"/api/previous?before={queue[2]['position']}").get_json()["item"]
    assert item["id"] == queue[1]["id"] and item["unidentified"]
    item = client.get(f"/api/previous?before={item['position']}").get_json()["item"]
    assert item["id"] == queue[0]["id"]
    assert client.get(f"/api/previous?before={item['position']}").get_json()["item"] is None


def test_admin_needs_the_key(cfg):
    application = server.create_app(cfg)
    client = application.test_client()
    key = application.config["ADMIN_KEY"]
    assert client.get("/api/admin/progress?key=wrong").status_code == 404
    assert client.get("/admin/responses.csv").status_code == 404
    login(client)
    image_id = client.get("/api/state").get_json()["queue"][0]["id"]
    answer(client, image_id, ["dog", "tree"])
    progress = client.get(f"/api/admin/progress?key={key}").get_json()["users"]
    assert progress[0]["done"] == 1
    text = client.get(f"/admin/responses.csv?key={key}").get_data(as_text=True)
    assert "dog|tree" in text and "algorithm" in text


def test_admin_tracks_logins_and_pause_blocks_access(cfg):
    application = server.create_app(cfg)
    client = application.test_client()
    key = application.config["ADMIN_KEY"]
    progress = lambda: client.get(f"/api/admin/progress?key={key}").get_json()["users"]
    assert progress()[0]["logins"] == 0 and not progress()[0]["online"]
    login(client)
    first = progress()[0]
    assert first["username"] == "Ann" and first["logins"] == 1 and first["online"]
    assert client.post("/api/admin/pause", json={"username": "ann", "paused": True}).status_code == 404
    url = f"/api/admin/pause?key={key}"
    client.post(url, json={"username": "ann", "paused": True})
    assert client.get("/api/state").status_code == 401
    assert login(client).status_code == 403
    client.post(url, json={"username": "ann", "paused": False})
    assert login(client).status_code == 200


def test_reset_archives_answers(cfg):
    application = server.create_app(cfg)
    client = application.test_client()
    key = application.config["ADMIN_KEY"]
    login(client)
    image_id = client.get("/api/state").get_json()["queue"][0]["id"]
    answer(client, image_id, ["dog"])
    client.post(f"/api/admin/reset?key={key}", json={"username": "ann"})
    assert client.get("/api/state").get_json()["done"] == 0
    db = server.connect(Path(cfg["data_dir"]) / "responses.sqlite")
    assert db.execute("SELECT COUNT(*) c FROM responses_archive").fetchone()["c"] == 1
    db.close()


def test_backup_and_per_user_csv(cfg):
    application = server.create_app(cfg)
    client = application.test_client()
    key = application.config["ADMIN_KEY"]
    login(client)
    image_id = client.get("/api/state").get_json()["queue"][0]["id"]
    answer(client, image_id, ["dog"])
    assert client.get(f"/admin/backup.sqlite?key={key}").data[:6] == b"SQLite"
    assert "ann" in client.get(f"/admin/responses.csv?key={key}&user=ann").get_data(as_text=True)
    assert "ann" not in client.get(f"/admin/responses.csv?key={key}&user=bob").get_data(as_text=True)


def test_changed_config_cannot_silently_reassign(cfg):
    server.create_app(cfg)
    with pytest.raises(SystemExit):
        server.create_app({**cfg, "usernames": ["ann", "bob", "dee"]})


def test_order_never_repeats_an_algorithm_or_thread_count():
    rows = [{"image_id": f"{i:012x}", "algorithm": f"a{i % 10}", "threads_target": str(1000 * (i % 20 + 1))}
            for i in range(5000)]
    cell = {r["image_id"]: (r["algorithm"], r["threads_target"]) for r in rows}
    tasks = server.build_assignments(rows, ["x", "y"], 2, 1)
    assert tasks["x"] != tasks["y"]
    for order in tasks.values():
        assert not any(server.clash(cell[a], cell[b]) for a, b in zip(order, order[1:]))


def test_consent_is_needed_once(cfg):
    client = server.create_app(cfg).test_client()
    login(client, agree=False)
    state = client.get("/api/state").get_json()
    assert not state["consented"] and not state["intro_seen"]
    assert answer(client, state["queue"][0]["id"]).status_code == 403
    assert client.post("/api/consent", json={"agreed": False, "name": "Ann"}).status_code == 400
    assert client.post("/api/consent", json={"agreed": True, "name": " "}).status_code == 400
    assert client.post("/api/consent", json={"agreed": True, "name": "Ann Lee"}).status_code == 200
    client.post("/api/intro", json={})
    state = client.get("/api/state").get_json()
    assert state["consented"] and state["intro_seen"]
    assert answer(client, state["queue"][0]["id"]).status_code == 200


def test_skipped_image_goes_to_the_back(client):
    login(client)
    ids = client.get("/api/map").get_json()["ids"]
    client.post("/api/skip", json={"id": ids[0]})
    data = client.get("/api/map").get_json()
    assert data["ids"][0] != ids[0] and ids[0] in data["ids"][-3:]
    assert data["status"][data["ids"].index(ids[0])] == "s"
    assert client.get("/api/state").get_json()["queue"][0]["id"] != ids[0]


def test_closed_page_requeues_the_open_image(client):
    login(client)
    ids = client.get("/api/map").get_json()["ids"]
    client.post("/api/view", json={"id": ids[0]})
    assert client.get("/api/state").get_json()["queue"][0]["id"] == ids[0]
    assert client.get("/api/state?resume=1").get_json()["queue"][0]["id"] != ids[0]


def test_answered_image_is_not_requeued(client):
    login(client)
    ids = client.get("/api/map").get_json()["ids"]
    client.post("/api/view", json={"id": ids[0]})
    answer(client, ids[0])
    client.get("/api/state?resume=1")
    assert client.get("/api/map").get_json()["ids"][0] == ids[0]


def test_item_lookup(client):
    login(client)
    ids = client.get("/api/map").get_json()["ids"]
    answer(client, ids[0], ["cat"])
    assert client.get(f"/api/item/{ids[0]}").get_json()["labels"] == ["cat"]
    assert not client.get(f"/api/item/{ids[1]}").get_json()["answered"]


def test_text_is_stored_as_typed(client):
    login(client)
    image_id = client.get("/api/state").get_json()["queue"][0]["id"]
    sentence = "  A batter in a Baseball field, batting a ball "
    assert answer(client, image_id, [sentence]).status_code == 200
    assert client.get(f"/api/item/{image_id}").get_json()["labels"] == [sentence.strip()]


def test_boxes_are_stored_and_validated(client):
    login(client)
    image_id = client.get("/api/state").get_json()["queue"][0]["id"]
    box = {"label": "dog", "x": 5, "y": 6, "w": 50, "h": 40}
    assert answer(client, image_id, ["dog"], boxes=[box]).status_code == 200
    assert client.get(f"/api/item/{image_id}").get_json()["boxes"] == [box]
    assert answer(client, image_id, ["dog"], boxes=[{**box, "label": "cat"}]).status_code == 400
    assert answer(client, image_id, ["dog"], boxes=[{**box, "w": 0}]).status_code == 400
    assert answer(client, image_id, ["dog"], boxes="x").status_code == 400


def test_scoring():
    result = score.score_answer(["sofa", "Dog", "tree"], {"dog", "couch", "tv"})
    assert (result["hits"], result["said"], result["truth"]) == (2, 2, 3)
    assert score.score_answer(["dog"], set())["false_alarm"] == 1
    assert score.score_answer([], {"dog"})["recognised"] == 0


def test_dataset_assignment_is_balanced():
    scenes = {f"{i:012d}.jpg": {"width": 640, "height": 480,
                                "counts": Counter(person=i % 2, dog=1)}
              for i in range(5000)}
    items = build_dataset.assign(scenes)
    cells = Counter((i["algorithm"], i["threads_target"]) for i in items)
    assert len(cells) == 200 and set(cells.values()) == {25}
    assert len({i["source"] for i in items}) == 5000
    assert len({i["image_id"] for i in items}) == 5000
    per_split = Counter((i["algorithm"], i["threads_target"], i["split"]) for i in items)
    assert set(per_split.values()) == {15, 5}
    assert items == build_dataset.assign(scenes)


def test_vocabulary_is_consistent():
    assert len(CLASSES) == len(set(CLASSES)) == 80
    assert set(SYNONYMS.values()) <= set(CLASSES)
    assert canonical("  Mobile   Phone ") == "cell phone"
    assert canonical("mountain") is None
    assert canonical("Red car") == "car"
    assert canonical("two dogs") == "dog"
    assert canonical("hot dog") == "hot dog"
    assert canonical("bus") == "bus"
    assert canonical("men") == "person"
    assert classes_in("a batter in a field, hitting a ball with a bat") == {"sports ball", "baseball bat"}
    assert classes_in("dog, cat, hot dog") == {"dog", "cat", "hot dog"}


def test_coco_names_match_the_dataset():
    annotations = Path(__file__).resolve().parent.parent / "dataset" / "annotations.json"
    if not annotations.exists():
        pytest.skip("dataset not downloaded")
    coco = json.loads(annotations.read_text(encoding="utf-8"))
    assert {c["name"] for c in coco["categories"]} == set(CLASSES)
