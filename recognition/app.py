#!/usr/bin/env python3
"""Human recognition study server.

Usage (from the project root):
    python recognition/app.py
    python recognition/app.py --config config/recognition.json --port 8765

Answers are written to <data_dir>/responses.sqlite as they arrive. Work
assignments are fixed on first start and stored there too, so restarting the
server never changes anyone's task.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import io
import json
import random
import re
import secrets
import socket
import sqlite3
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path

from flask import (Flask, Response, abort, g, jsonify, request, send_file,
                   send_from_directory, session)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ROOT = HERE.parent
DEFAULT_CONFIG = ROOT / "config" / "recognition.json"

MAX_LABELS = 20
MAX_LABEL_LEN = 500
MAX_BOXES = 50
QUEUE_SIZE = 12

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS assignments (
    username TEXT NOT NULL,
    position REAL NOT NULL,
    image_id TEXT NOT NULL,
    skipped INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (username, position),
    UNIQUE (username, image_id)
);
CREATE TABLE IF NOT EXISTS responses (
    username TEXT NOT NULL,
    image_id TEXT NOT NULL,
    labels TEXT NOT NULL,
    boxes TEXT NOT NULL DEFAULT '[]',
    unidentified INTEGER NOT NULL,
    response_ms INTEGER,
    shown_w INTEGER,
    shown_h INTEGER,
    dpr REAL,
    device TEXT,
    edits INTEGER NOT NULL DEFAULT 0,
    answered_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    PRIMARY KEY (username, image_id)
);
CREATE TABLE IF NOT EXISTS responses_archive (
    username TEXT, image_id TEXT, labels TEXT, boxes TEXT, unidentified INTEGER,
    response_ms INTEGER, shown_w INTEGER, shown_h INTEGER, dpr REAL,
    device TEXT, edits INTEGER, answered_at TEXT,
    reset_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE TABLE IF NOT EXISTS accounts (
    username TEXT PRIMARY KEY,
    first_login TEXT,
    last_login TEXT,
    logins INTEGER NOT NULL DEFAULT 0,
    last_seen TEXT,
    paused INTEGER NOT NULL DEFAULT 0,
    consent_name TEXT,
    consent_at TEXT,
    intro_seen INTEGER NOT NULL DEFAULT 0,
    viewing_id TEXT
);
"""

NOW = "strftime('%Y-%m-%dT%H:%M:%SZ', 'now')"
ONLINE_SECONDS = 120
TAIL_SEARCH = 40  # how far from the end a requeued image may land


def load_manifest(data_dir: Path) -> list[dict]:
    with open(data_dir / "manifest.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clash(a: tuple, b: tuple) -> bool:
    """True if two images share an algorithm or a thread count."""
    return a[0] == b[0] or a[1] == b[1]


def spaced_order(ids: list[str], cell: dict, rng: random.Random) -> list[str]:
    """Shuffle so neighbours never share an algorithm or thread count."""
    best, best_bad = list(ids), len(ids) + 1
    for _ in range(20):
        order = list(ids)
        rng.shuffle(order)
        for i in range(1, len(order)):
            before = cell[order[i - 1]]
            if not clash(before, cell[order[i]]):
                continue
            for j in range(i + 1, len(order)):
                if not clash(before, cell[order[j]]):
                    order[i], order[j] = order[j], order[i]
                    break
            else:  # nothing left ahead: swap into an earlier gap
                item = order[i]
                after = cell[order[i + 1]] if i + 1 < len(order) else None
                for k in range(1, i - 1):
                    other = cell[order[k]]
                    if (not clash(cell[order[k - 1]], cell[item])
                            and not clash(cell[item], cell[order[k + 1]])
                            and not clash(before, other)
                            and (after is None or not clash(other, after))):
                        order[i], order[k] = order[k], order[i]
                        break
        bad = sum(clash(cell[a], cell[b]) for a, b in zip(order, order[1:]))
        if bad < best_bad:
            best, best_bad = order, bad
        if not bad:
            break
    return best


def build_assignments(rows: list[dict], users: list[str], raters: int,
                      seed: int) -> dict[str, list[str]]:
    """Give each user an equal, cell-balanced share in their own spaced order."""
    ordered = sorted(rows, key=lambda r: (r["algorithm"], int(r["threads_target"]),
                                          r["image_id"]))
    tasks: dict[str, list[str]] = {u: [] for u in users}
    for i, row in enumerate(ordered):
        for k in range(raters):
            tasks[users[(i + k) % len(users)]].append(row["image_id"])
    cell = {r["image_id"]: (r["algorithm"], r["threads_target"]) for r in rows}
    return {user: spaced_order(ids, cell, random.Random(f"{seed}:{user}"))
            for user, ids in tasks.items()}


def dataset_hash(rows: list[dict]) -> str:
    ids = sorted(f"{r['image_id']}:{r['path']}" for r in rows)
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    return db


def meta_value(db: sqlite3.Connection, key: str, default=None):
    """Return a meta value, storing `default` on first use."""
    row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    if row:
        return row["value"]
    db.execute("INSERT INTO meta (key, value) VALUES (?, ?)", (key, default))
    return default


def initialise(db_path: Path, cfg: dict, rows: list[dict]) -> tuple[str, str]:
    """Create tables and assignments. Return (cookie secret, admin key)."""
    users = [u.lower() for u in cfg["usernames"]]
    raters = cfg["raters_per_image"]
    if len(set(users)) != len(users) or not users:
        sys.exit("config: usernames must be unique and not empty")
    if not 1 <= raters <= len(users):
        sys.exit("config: raters_per_image must be between 1 and the number of users")

    db = connect(db_path)
    with db:
        db.executescript(SCHEMA)
        fingerprint = json.dumps({"users": users, "raters": raters,
                                  "seed": cfg["order_seed"],
                                  "images": dataset_hash(rows)})
        stored = meta_value(db, "assignment", fingerprint)
        if stored != fingerprint:
            sys.exit("config: usernames, raters, seed or images changed since the "
                     "study started. Restore them, or use a new data folder.")
        if not db.execute("SELECT 1 FROM assignments LIMIT 1").fetchone():
            tasks = build_assignments(rows, users, raters, cfg["order_seed"])
            db.executemany(
                "INSERT INTO assignments (username, position, image_id) "
                "VALUES (?, ?, ?)",
                ((u, p, i) for u, ids in tasks.items() for p, i in enumerate(ids)))
        db.executemany("INSERT OR IGNORE INTO accounts (username) VALUES (?)",
                       ((u,) for u in users))
        secret = meta_value(db, "secret", secrets.token_hex(32))
        admin = meta_value(db, "admin_key", secrets.token_urlsafe(18))
    db.close()
    return secret, admin


def clean_labels(raw) -> list[str] | None:
    if not isinstance(raw, list) or len(raw) > MAX_LABELS:
        return None
    labels: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            return None
        text = item.strip()[:MAX_LABEL_LEN]  # kept as typed, for later analysis
        if text and text not in labels:
            labels.append(text)
    return labels


def clean_boxes(raw, labels: list[str], width: int, height: int) -> list[dict] | None:
    """Validate boxes given in image pixels; each must name one of the labels."""
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) > MAX_BOXES:
        return None
    boxes = []
    for item in raw:
        if not isinstance(item, dict) or item.get("label") not in labels:
            return None
        x = number(item.get("x"), 0, width)
        y = number(item.get("y"), 0, height)
        w = number(item.get("w"), 1, width)
        h = number(item.get("h"), 1, height)
        if None in (x, y, w, h) or x + w > width or y + h > height:
            return None
        boxes.append({"label": item["label"], "x": x, "y": y, "w": w, "h": h})
    return boxes


def number(value, low, high, kind=int):
    """Coerce client telemetry; bad values become None, never an error."""
    try:
        result = kind(value)
    except (TypeError, ValueError):
        return None
    return result if low <= result <= high else None


def export_csv(db: sqlite3.Connection, rows: list[dict],
               user: str | None = None) -> str:
    by_id = {r["image_id"]: r for r in rows}
    columns = ["username", "image_id", "algorithm", "threads_target",
               "threads_actual", "source", "split", "classes", "labels",
               "boxes",
               "unidentified", "response_ms", "shown_w", "shown_h", "dpr", "device",
               "edits", "skipped", "answered_at"]
    out = io.StringIO()
    writer = csv.DictWriter(out, columns, extrasaction="ignore")
    writer.writeheader()
    query = """SELECT r.*, a.skipped FROM responses r JOIN assignments a
               ON a.username = r.username AND a.image_id = r.image_id
               WHERE ? IS NULL OR r.username = ?
               ORDER BY r.username, r.answered_at"""
    for r in db.execute(query, (user, user)):
        row = {**by_id.get(r["image_id"], {}), **dict(r)}
        row["labels"] = "|".join(json.loads(r["labels"]))
        writer.writerow(row)
    return out.getvalue()


def create_app(cfg: dict) -> Flask:
    data_dir = Path(cfg["data_dir"])
    db_path = Path(cfg.get("db_path") or data_dir / "responses.sqlite")
    rows = load_manifest(data_dir)
    paths = {r["image_id"]: data_dir / r["path"] for r in rows}
    cells = {r["image_id"]: (r["algorithm"], r["threads_target"]) for r in rows}
    sizes = {r["image_id"]: (int(r.get("width") or 100_000),
                             int(r.get("height") or 100_000)) for r in rows}
    users = {u.lower(): u for u in cfg["usernames"]}  # login ignores case
    secret, admin_key = initialise(db_path, cfg, rows)
    with connect(db_path) as conn:
        paused = {r["username"] for r in
                  conn.execute("SELECT username FROM accounts WHERE paused = 1")}

    app = Flask(__name__, static_folder=None)
    app.secret_key = secret
    app.json.sort_keys = False  # label groups keep their order
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        PERMANENT_SESSION_LIFETIME=timedelta(days=90),
        MAX_CONTENT_LENGTH=64 * 1024,
        ADMIN_KEY=admin_key,
    )

    def db() -> sqlite3.Connection:
        if "db" not in g:
            g.db = connect(db_path)
        return g.db

    @app.teardown_appcontext
    def close_db(_exc):
        conn = g.pop("db", None)
        if conn is not None:
            conn.close()

    def current_user() -> str:
        user = session.get("user")
        if user not in users or user in paused:
            abort(401)
        return user

    def seen(conn: sqlite3.Connection, user: str) -> None:
        conn.execute(f"UPDATE accounts SET last_seen = {NOW} WHERE username = ?",
                     (user,))

    def counts(user: str) -> tuple[int, int]:
        total = db().execute("SELECT COUNT(*) c FROM assignments WHERE username=?",
                             (user,)).fetchone()["c"]
        done = db().execute("SELECT COUNT(*) c FROM responses WHERE username=?",
                            (user,)).fetchone()["c"]
        return total, done

    def next_items(user: str, after: int) -> list[dict]:
        query = """SELECT a.position, a.image_id FROM assignments a
                   LEFT JOIN responses r
                     ON r.username = a.username AND r.image_id = a.image_id
                   WHERE a.username = ? AND r.image_id IS NULL AND a.position > ?
                   ORDER BY a.position LIMIT ?"""
        found = db().execute(query, (user, after, QUEUE_SIZE)).fetchall()
        return [{"position": r["position"], "id": r["image_id"]} for r in found]

    def move_to_back(conn: sqlite3.Connection, user: str, image_id: str) -> None:
        """Requeue an image at the end, still avoiding a matching neighbour."""
        tail = conn.execute(
            """SELECT position, image_id FROM assignments
               WHERE username = ? AND image_id != ?
               ORDER BY position DESC LIMIT ?""",
            (user, image_id, TAIL_SEARCH)).fetchall()
        if not tail:
            return
        cell = cells[image_id]
        position = tail[0]["position"] + 1
        if clash(cell, cells[tail[0]["image_id"]]):
            for later, earlier in zip(tail, tail[1:]):
                if not clash(cell, cells[later["image_id"]]) and \
                        not clash(cell, cells[earlier["image_id"]]):
                    position = (later["position"] + earlier["position"]) / 2
                    break
        conn.execute(
            """UPDATE assignments SET position = ?, skipped = skipped + 1
               WHERE username = ? AND image_id = ?""", (position, user, image_id))

    def answered(conn: sqlite3.Connection, user: str, image_id: str) -> bool:
        return conn.execute(
            "SELECT 1 FROM responses WHERE username = ? AND image_id = ?",
            (user, image_id)).fetchone() is not None

    def assigned(conn: sqlite3.Connection, user: str, image_id: str) -> bool:
        return conn.execute(
            "SELECT 1 FROM assignments WHERE username = ? AND image_id = ?",
            (user, image_id)).fetchone() is not None

    def account(user: str) -> sqlite3.Row:
        return db().execute("SELECT * FROM accounts WHERE username = ?",
                            (user,)).fetchone()

    def pace_ms(user: str) -> int | None:
        found = db().execute(
            "SELECT response_ms FROM responses WHERE username=? AND response_ms "
            "IS NOT NULL ORDER BY answered_at DESC LIMIT 50", (user,)).fetchall()
        if len(found) < 5:
            return None
        return sorted(r["response_ms"] for r in found)[len(found) // 2]

    # Pages ------------------------------------------------------------------

    @app.get("/")
    def index():
        return send_from_directory(HERE / "static", "index.html", max_age=0)

    @app.get("/static/<path:name>")
    def static_file(name):
        return send_from_directory(HERE / "static", name, max_age=0)

    # Session ----------------------------------------------------------------

    @app.post("/api/login")
    def login():
        body = request.get_json(silent=True) or {}
        name = str(body.get("username", "")).strip().lower()
        if name not in users:
            return jsonify(error="Unknown username"), 401
        if name in paused:
            return jsonify(error="Account paused"), 403
        session.clear()
        session.permanent = True
        session["user"] = name
        with db() as conn:
            conn.execute(
                f"""UPDATE accounts SET logins = logins + 1, last_login = {NOW},
                        last_seen = {NOW}, first_login = COALESCE(first_login, {NOW})
                    WHERE username = ?""", (name,))
        return jsonify(ok=True)

    @app.post("/api/logout")
    def logout():
        session.clear()
        return jsonify(ok=True)

    # Task -------------------------------------------------------------------

    @app.get("/api/state")
    def state():
        user = current_user()
        with db() as conn:
            seen(conn, user)
            viewing = account(user)["viewing_id"]
            if request.args.get("resume") and viewing:
                # The page was closed on this image: it goes to the back.
                if not answered(conn, user, viewing):
                    move_to_back(conn, user, viewing)
                conn.execute("UPDATE accounts SET viewing_id = NULL WHERE username = ?",
                             (user,))
        total, done = counts(user)
        row = account(user)
        return jsonify(username=users[user], total=total, done=done,
                       queue=next_items(user, -1), pace_ms=pace_ms(user),
                       consented=row["consent_at"] is not None,
                       intro_seen=bool(row["intro_seen"]),
                       contact=cfg.get("contact", ""))

    @app.post("/api/consent")
    def consent():
        user = current_user()
        body = request.get_json(silent=True) or {}
        name = " ".join(str(body.get("name", "")).split())[:120]
        if body.get("agreed") is not True or len(name) < 2:
            return jsonify(error="Tick the box and enter your full name"), 400
        with db() as conn:
            conn.execute(
                f"""UPDATE accounts SET consent_name = ?, consent_at = {NOW}
                    WHERE username = ? AND consent_at IS NULL""", (name, user))
        return jsonify(ok=True)

    @app.post("/api/intro")
    def intro_seen():
        user = current_user()
        with db() as conn:
            conn.execute("UPDATE accounts SET intro_seen = 1 WHERE username = ?", (user,))
        return jsonify(ok=True)

    @app.post("/api/view")
    def view():
        """Remember the open image, so closing the page can requeue it."""
        user = current_user()
        image_id = str((request.get_json(silent=True) or {}).get("id", ""))
        with db() as conn:
            if assigned(conn, user, image_id):
                conn.execute("UPDATE accounts SET viewing_id = ? WHERE username = ?",
                             (image_id, user))
        return jsonify(ok=True)

    @app.post("/api/skip")
    def skip():
        user = current_user()
        image_id = str((request.get_json(silent=True) or {}).get("id", ""))
        with db() as conn:
            if not assigned(conn, user, image_id):
                return jsonify(error="Image not assigned"), 403
            if not answered(conn, user, image_id):
                move_to_back(conn, user, image_id)
            conn.execute(
                "UPDATE accounts SET viewing_id = NULL "
                "WHERE username = ? AND viewing_id = ?", (user, image_id))
        return jsonify(ok=True)

    @app.get("/api/map")
    def task_map():
        """Every task in queue order: d done, u unidentified, s skipped, r remaining."""
        user = current_user()
        found = db().execute(
            """SELECT a.image_id, a.skipped, r.unidentified FROM assignments a
               LEFT JOIN responses r
                 ON r.username = a.username AND r.image_id = a.image_id
               WHERE a.username = ? ORDER BY a.position""", (user,)).fetchall()
        def status(row):
            if row["unidentified"] is None:
                return "s" if row["skipped"] else "r"
            return "u" if row["unidentified"] else "d"
        return jsonify(ids=[r["image_id"] for r in found],
                       status="".join(status(r) for r in found))

    @app.get("/api/item/<image_id>")
    def task_item(image_id):
        user = current_user()
        row = db().execute(
            """SELECT a.position, a.image_id, r.labels, r.boxes, r.unidentified
               FROM assignments a
               LEFT JOIN responses r
                 ON r.username = a.username AND r.image_id = a.image_id
               WHERE a.username = ? AND a.image_id = ?""", (user, image_id)).fetchone()
        if row is None:
            abort(404)
        return jsonify(position=row["position"], id=row["image_id"],
                       answered=row["labels"] is not None,
                       labels=json.loads(row["labels"] or "[]"),
                       boxes=json.loads(row["boxes"] or "[]"),
                       unidentified=bool(row["unidentified"]))

    @app.get("/api/next")
    def next_batch():
        user = current_user()
        after = number(request.args.get("after"), -1, 10**7, float)
        return jsonify(queue=next_items(user, -1 if after is None else after))

    @app.get("/api/previous")
    def previous():
        """Latest answered image before a position; powers the Back button."""
        user = current_user()
        before = number(request.args.get("before"), 0, 10**7, float)
        if before is None:
            before = 10**7
        row = db().execute(
            """SELECT a.position, a.image_id, r.labels, r.boxes, r.unidentified
               FROM assignments a
               JOIN responses r ON r.username = a.username AND r.image_id = a.image_id
               WHERE a.username = ? AND a.position < ?
               ORDER BY a.position DESC LIMIT 1""", (user, before)).fetchone()
        if row is None:
            return jsonify(item=None)
        return jsonify(item={"position": row["position"], "id": row["image_id"],
                             "labels": json.loads(row["labels"]),
                             "boxes": json.loads(row["boxes"]),
                             "unidentified": bool(row["unidentified"])})

    @app.post("/api/answer")
    def answer():
        user = current_user()
        body = request.get_json(silent=True) or {}
        image_id = str(body.get("id", ""))
        labels = clean_labels(body.get("labels"))
        unidentified = bool(body.get("unidentified"))
        if labels is None or (not labels and not unidentified) or (labels and unidentified):
            return jsonify(error="Invalid answer"), 400
        boxes = clean_boxes(body.get("boxes"), labels or [],
                            *sizes.get(image_id, (1, 1)))
        if boxes is None:
            return jsonify(error="Invalid box"), 400
        conn = db()
        if account(user)["consent_at"] is None:
            return jsonify(error="Consent required"), 403
        if not assigned(conn, user, image_id):
            return jsonify(error="Image not assigned"), 403
        with conn:
            seen(conn, user)
            conn.execute(
                """INSERT INTO responses (username, image_id, labels, boxes,
                       unidentified, response_ms, shown_w, shown_h, dpr, device)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (username, image_id) DO UPDATE SET
                       edits = edits + (labels <> excluded.labels
                                        OR boxes <> excluded.boxes),
                       labels = excluded.labels, boxes = excluded.boxes,
                       unidentified = excluded.unidentified""",
                (user, image_id, json.dumps(labels), json.dumps(boxes),
                 int(unidentified),
                 number(body.get("ms"), 0, 3_600_000),
                 number(body.get("w"), 1, 20_000),
                 number(body.get("h"), 1, 20_000),
                 number(body.get("dpr"), 0.25, 10, float),
                 "phone" if body.get("touch") else "desktop"))
        return jsonify(ok=True, done=counts(user)[1])

    @app.get("/img/<image_id>")
    def image(image_id):
        current_user()
        path = paths.get(image_id) if re.fullmatch(r"[0-9a-f]{12}", image_id) else None
        if path is None or not path.exists():
            abort(404)
        response = send_file(path, mimetype="image/png", etag=False,
                             last_modified=None, conditional=False)
        response.headers.pop("Content-Disposition", None)  # names the source scene
        response.headers["Cache-Control"] = "private, max-age=604800, immutable"
        return response

    # Admin ------------------------------------------------------------------

    def require_admin():
        if not hmac.compare_digest(request.args.get("key", ""), admin_key):
            abort(404)

    @app.get("/admin")
    def admin():
        require_admin()
        return send_from_directory(HERE / "static", "admin.html", max_age=0)

    @app.get("/api/admin/progress")
    def admin_progress():
        require_admin()
        result = []
        now = db().execute(f"SELECT {NOW} t").fetchone()["t"]
        for name in cfg["usernames"]:
            user = name.lower()
            total, done = counts(user)
            stats = db().execute(
                """SELECT COALESCE(SUM(unidentified), 0) unidentified,
                          COALESCE(SUM(response_ms), 0) spent_ms,
                          COALESCE(SUM(edits), 0) edits FROM responses
                   WHERE username = ?""", (user,)).fetchone()
            skipped = db().execute(
                """SELECT COUNT(*) n FROM assignments a WHERE a.username = ?
                   AND a.skipped > 0 AND NOT EXISTS (SELECT 1 FROM responses r
                       WHERE r.username = a.username AND r.image_id = a.image_id)""",
                (user,)).fetchone()["n"]
            device = db().execute(
                "SELECT device FROM responses WHERE username = ? "
                "ORDER BY answered_at DESC LIMIT 1", (user,)).fetchone()
            info = db().execute(
                "SELECT *, strftime('%s', ?) - strftime('%s', last_seen) idle "
                "FROM accounts WHERE username = ?", (now, user)).fetchone()
            result.append({
                "username": name, "total": total, "done": done,
                "unidentified": stats["unidentified"], "edits": stats["edits"],
                "spent_ms": stats["spent_ms"], "pace_ms": pace_ms(user),
                "skipped": skipped, "consent_name": info["consent_name"],
                "consent_at": info["consent_at"],
                "device": device["device"] if device else None,
                "logins": info["logins"], "first_login": info["first_login"],
                "last_seen": info["last_seen"], "paused": bool(info["paused"]),
                "online": info["idle"] is not None
                          and info["idle"] <= ONLINE_SECONDS})
        return jsonify(users=result)

    def admin_target() -> str:
        body = request.get_json(silent=True) or {}
        user = str(body.get("username", "")).lower()
        if user not in users:
            abort(400)
        return user

    @app.post("/api/admin/pause")
    def admin_pause():
        require_admin()
        user = admin_target()
        value = bool((request.get_json(silent=True) or {}).get("paused"))
        with db() as conn:
            conn.execute("UPDATE accounts SET paused = ? WHERE username = ?",
                         (int(value), user))
        (paused.add if value else paused.discard)(user)
        return jsonify(ok=True)

    @app.post("/api/admin/reset")
    def admin_reset():
        """Clear a user's answers. They move to responses_archive, not the bin."""
        require_admin()
        user = admin_target()
        with db() as conn:
            conn.execute(
                """INSERT INTO responses_archive (username, image_id, labels, boxes,
                       unidentified, response_ms, shown_w, shown_h, dpr, device,
                       edits, answered_at)
                   SELECT username, image_id, labels, boxes, unidentified, response_ms,
                       shown_w, shown_h, dpr, device, edits, answered_at
                   FROM responses WHERE username = ?""", (user,))
            conn.execute("DELETE FROM responses WHERE username = ?", (user,))
        return jsonify(ok=True)

    @app.get("/admin/responses.csv")
    def admin_csv():
        require_admin()
        user = request.args.get("user", "").lower()
        if user and user not in users:
            abort(404)
        name = f"responses_{users[user]}.csv" if user else "responses.csv"
        return Response(export_csv(db(), rows, user or None), mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={name}"})

    @app.get("/admin/consents.csv")
    def admin_consents():
        require_admin()
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["username", "full_name", "consented_at"])
        for name in cfg["usernames"]:
            row = account(name.lower())
            if row["consent_at"]:
                writer.writerow([name, row["consent_name"], row["consent_at"]])
        return Response(out.getvalue(), mimetype="text/csv", headers={
            "Content-Disposition": "attachment; filename=consents.csv"})

    @app.get("/admin/backup.sqlite")
    def admin_backup():
        require_admin()
        with tempfile.TemporaryDirectory() as tmp:
            copy = sqlite3.connect(Path(tmp) / "backup.sqlite")
            db().backup(copy)
            copy.close()
            data = (Path(tmp) / "backup.sqlite").read_bytes()
        return Response(data, mimetype="application/vnd.sqlite3", headers={
            "Content-Disposition": "attachment; filename=responses_backup.sqlite"})

    return app


def lan_address() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def main() -> None:
    ap = argparse.ArgumentParser(description="Human recognition study server")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--port", type=int)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    port = args.port or cfg.get("port", 8765)
    app = create_app(cfg)

    print(f"Local:  http://localhost:{port}")
    print(f"LAN:    http://{lan_address()}:{port}")
    print(f"Admin:  http://localhost:{port}/admin?key={app.config['ADMIN_KEY']}")
    print("Started", time.strftime("%H:%M:%S"), flush=True)

    from waitress import serve
    serve(app, host="0.0.0.0", port=port, threads=8)


if __name__ == "__main__":
    main()
