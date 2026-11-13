"use strict";

const $ = (id) => document.getElementById(id);
const touch = matchMedia("(pointer: coarse)").matches;

const PENDING_KEY = "recognition.pending";
const PRELOAD = 5;
const LAST_POSITION = 10000000;
const MIN_SHOW_MS = 250; // ignores a double tap landing on the next image
const COLORS = { d: "--done", u: "--flag", s: "--skip", r: "--line" };

let total = 0;
let done = 0;
let paceMs = null;
let username = "";
let introSeen = false;

let queue = [];       // upcoming {position, id}
let current = null;   // {position, id, saved}
let picked = [];
let boxes = [];       // saved with the answer, in image pixels
let pendingBox = null;
let draft = null;
let shownAt = 0;
let hiddenMs = 0;
let hiddenSince = null;
let pending = readPending();
let flushRun = null;
let retryMs = 2000;
let fetching = false;
let taskMap = { ids: [], status: "" };
const recent = [];    // response times this session

/* Storage is a convenience: unsent answers survive a closed tab where it works. */
function readPending() {
  try { return JSON.parse(localStorage.getItem(PENDING_KEY)) || []; }
  catch { return []; }
}
function writePending() {
  try { localStorage.setItem(PENDING_KEY, JSON.stringify(pending)); }
  catch { /* private mode: answers still send while the tab is open */ }
}

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    keepalive: true,
  };
  const res = await fetch(path, options);
  if (res.status === 401) {
    const err = new Error("signed out");
    err.signedOut = true;
    throw err;
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data.error || res.statusText);
    err.rejected = res.status >= 400 && res.status < 500;
    throw err;
  }
  return data;
}

function show(view) {
  for (const id of ["login", "consent", "intro", "task", "finished"]) $(id).hidden = id !== view;
  $("sheet").hidden = true;
  if (view === "intro") window.tutorial.play(!introSeen); else window.tutorial.stop();
  if (view === "login") $("username").focus();
  if (view === "task") {
    paintBar();
    if (!touch) $("entry").focus();
  }
}

/* Theme ------------------------------------------------------------------ */

function paintTheme() {
  const dark = document.documentElement.dataset.theme !== "light";
  $("theme").textContent = dark ? "Light mode" : "Dark mode";
}

$("theme").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("recognition.theme", next); } catch { /* this visit only */ }
  paintTheme();
  paintBar();
});
paintTheme();

/* Sign-in, consent, intro ------------------------------------------------- */

$("login").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("login-error").textContent = "";
  try {
    await api("/api/login", { username: $("username").value });
    await start();
  } catch (err) {
    $("login-error").textContent = err.signedOut ? "Unknown username"
      : err.rejected ? err.message : "Connection problem";
  }
});

$("consent").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("consent-error").textContent = "";
  try {
    await api("/api/consent", { agreed: $("agree").checked, name: $("fullname").value });
    await start();
  } catch (err) {
    $("consent-error").textContent = err.rejected ? err.message : "Connection problem";
  }
});

function leaveIntro() {
  if (!introSeen) {
    introSeen = true;
    api("/api/intro", {}).catch(() => {});
  }
  if (current || queue.length) {
    show("task");
    if (!current) advance();
  } else {
    show("finished");
  }
}
$("intro-start").addEventListener("click", leaveIntro);
$("help").addEventListener("click", () => show("intro"));

async function start() {
  await flush();
  let state;
  try {
    state = await api("/api/state?resume=1");
  } catch (err) {
    if (!err.signedOut) $("login-error").textContent = "Connection problem";
    return show("login");
  }
  ({ total, done, username } = state);
  paceMs = state.pace_ms;
  introSeen = state.intro_seen;
  $("who").textContent = `Signed in as ${username}`;
  loadSize();
  $("contact").textContent = state.contact ? `Questions: ${state.contact}` : "";
  if (!state.consented) return show("consent");
  queue = state.queue;
  await loadMap();
  if (!introSeen) {
    show("intro");
    return;
  }
  show("task");
  advance();
}

/* Moving through images --------------------------------------------------- */

function preload() {
  for (const item of queue.slice(0, PRELOAD)) new Image().src = `/img/${item.id}`;
}

async function topUp() {
  if (fetching || queue.length > PRELOAD) return;
  const last = queue.length ? queue[queue.length - 1].position : current ? current.position : -1;
  fetching = true;
  try {
    const { queue: more } = await api(`/api/next?after=${last}`);
    const known = new Set(queue.map((i) => i.id));
    if (current) known.add(current.id);
    for (const a of pending) known.add(a.id);
    queue.push(...more.filter((i) => !known.has(i.id)));
    preload();
  } catch { /* retried on the next answer */ }
  fetching = false;
}

async function advance() {
  if (!queue.length) {
    await flush();
    await topUp();
    if (!queue.length) {
      try {
        const state = await api("/api/state");
        done = state.done;
        queue = state.queue.filter((i) => !pending.some((a) => a.id === i.id));
      } catch { /* offline: stays on the current screen */ }
    }
    if (!queue.length) {
      if (pending.length) return void setTimeout(advance, 1000);
      return show("finished");
    }
  }
  const item = queue.shift();
  present(item, item.labels || [], item.saved ? item.saved : false, item.boxes || []);
  topUp();
  preload();
}

/* item.saved marks an image that already has a stored answer. */
function present(item, labels, saved, savedBoxes = []) {
  current = { ...item, saved };
  picked = labels.slice();
  boxes = savedBoxes.slice();
  closeChooser();
  $("entry").value = "";
  renderObjects();
  renderCounter();
  $("back").disabled = done === 0;
  $("status").textContent = saved
    ? (saved === "unidentified" ? "Saved as Unidentified. Change it or press Enter." : "Saved answer. Change it or press Enter.")
    : "";
  const img = $("image");
  img.classList.add("loading");
  shownAt = 0;
  img.dataset.id = item.id;
  img.src = `/img/${item.id}`;
  $("boxes").replaceChildren();
  if (!saved) api("/api/view", { id: item.id }).catch(() => {});
  if (!touch) $("entry").focus();
}

$("image").addEventListener("load", async (event) => {
  const img = event.target;
  const id = img.dataset.id;
  await img.decode().catch(() => {});
  if (!current || current.id !== id) return;
  img.classList.remove("loading");
  fitOverlay();
  renderBoxes();
  shownAt = performance.now();
  hiddenMs = 0;
  hiddenSince = document.hidden ? shownAt : null;
});

$("image").addEventListener("error", (event) => {
  const id = event.target.dataset.id;
  setTimeout(() => {
    if (current && current.id === id) event.target.src = `/img/${id}?r=${Date.now()}`;
  }, 2000);
});

document.addEventListener("visibilitychange", () => {
  const now = performance.now();
  if (document.hidden) hiddenSince = now;
  else if (hiddenSince !== null) { hiddenMs += now - hiddenSince; hiddenSince = null; }
  if (document.hidden) flush();
});

function ready() {
  return current && shownAt && performance.now() - shownAt >= MIN_SHOW_MS;
}

function save() {
  const img = $("image");
  const ms = Math.round(performance.now() - shownAt - hiddenMs);
  pending = pending.filter((a) => a.id !== current.id);
  pending.push({
    id: current.id,
    labels: picked,
    boxes,
    unidentified: picked.length === 0,
    ms,
    w: img.clientWidth,
    h: img.clientHeight,
    dpr: window.devicePixelRatio,
    touch,
  });
  writePending();
  if (!current.saved) {
    done += 1;
    recent.push(ms);
    if (recent.length > 50) recent.shift();
  }
  mark(current.id, picked.length ? "d" : "u");
  current = null;
  flush();
  advance();
}

function unidentified() {
  if (!ready()) return;
  picked = [];
  boxes = [];
  $("entry").value = "";
  save();
}

/* Next: saves what is entered. An untouched new image is skipped to the back. */
function next() {
  if (!ready()) return;
  addTyped();
  if (picked.length || current.saved) return save();
  const id = current.id;
  moveToEnd(id);
  current = null;
  api("/api/skip", { id }).then(() => { if (!queue.length) advance(); }).catch(() => {});
  if (queue.length) advance();
}

async function goBack(fromEnd = false) {
  if (!current && !fromEnd) return;
  $("back").disabled = true;
  addTyped();
  await flush();
  if (pending.length) return void ($("back").disabled = false);
  try {
    const before = current ? current.position : LAST_POSITION;
    const { item } = await api(`/api/previous?before=${before}`);
    if (!item) return;
    hold();
    show("task");
    present(item, item.labels, item.unidentified ? "unidentified" : "objects", item.boxes);
  } catch { $("back").disabled = false; }
}

/* Keeps the open image and its typed objects next in line. */
function hold() {
  if (current) queue.unshift({ position: current.position, id: current.id, labels: picked, boxes, saved: current.saved });
}

async function jumpTo(id) {
  if (current && current.id === id) return closeSheet();
  addTyped();
  await flush();
  try {
    const item = await api(`/api/item/${id}`);
    hold();
    queue = queue.filter((i) => i.id !== id);
    closeSheet();
    show("task");
    const saved = item.answered ? (item.unidentified ? "unidentified" : "objects") : false;
    present({ position: item.position, id }, item.labels, saved, item.boxes);
  } catch { /* the grid stays open */ }
}

$("next").addEventListener("click", next);
$("unidentified").addEventListener("click", unidentified);
$("back").addEventListener("click", () => goBack());
$("review").addEventListener("click", () => goBack(true));

/* Saving ------------------------------------------------------------------ */

/* Sends queued answers in order. Callers can await whichever run is active. */
function flush() {
  if (!flushRun) flushRun = drain().finally(() => { flushRun = null; });
  return flushRun;
}

async function drain() {
  while (pending.length) {
    const answer = pending[0];
    try {
      const result = await api("/api/answer", answer);
      if (pending[0] === answer) pending.shift();
      writePending();
      done = Math.max(done, result.done);
      retryMs = 2000;
      $("sync").textContent = "";
      renderCounter();
    } catch (err) {
      if (err.signedOut) return show("login");
      if (err.rejected) { // never block the queue on an answer the server refuses
        pending.shift();
        writePending();
        continue;
      }
      $("sync").textContent = "Not connected. Answers will send when back online.";
      setTimeout(flush, retryMs);
      retryMs = Math.min(retryMs * 2, 30000);
      return;
    }
  }
}

window.addEventListener("online", flush);

/* Objects ----------------------------------------------------------------- */

function addTyped() {
  const text = $("entry").value.trim();
  $("entry").value = "";
  if (text && !picked.includes(text)) picked.push(text);
  if (text && pendingBox) mapBox(text);
  renderObjects();
}

function renderObjects() {
  $("objects").replaceChildren(...picked.map((label) => {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.textContent = label;
    const count = boxes.filter((b) => b.label === label).length;
    const tag = document.createElement("em");
    tag.textContent = count ? `${count} box${count > 1 ? "es" : ""}` : "";
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Remove ${label}`);
    remove.onclick = () => {
      picked = picked.filter((p) => p !== label);
      boxes = boxes.filter((b) => b.label !== label);
      renderBoxes();
      renderObjects();
      if (!touch) $("entry").focus();
    };
    li.append(name, tag, remove);
    return li;
  }));
  $("none").hidden = picked.length > 0;
}

$("entry").addEventListener("keydown", (event) => {
  const empty = !event.target.value;
  if (event.key === "Enter") {
    event.preventDefault();
    if (!empty && event.target.value.trim()) return addTyped();
    if (event.repeat) return;
    if (picked.length) { if (ready()) save(); } else { unidentified(); }
  } else if (event.key === "Backspace" && empty && picked.length) {
    const label = picked.pop();
    boxes = boxes.filter((b) => b.label !== label);
    renderBoxes();
    renderObjects();
  } else if (event.key === "Escape" && pendingBox) {
    closeChooser();
    renderBoxes();
  } else if (/^[1-9]$/.test(event.key) && empty && pendingBox && picked[Number(event.key) - 1]) {
    event.preventDefault();
    mapBox(picked[Number(event.key) - 1]);
  } else if (event.key === "ArrowLeft" && empty) {
    event.preventDefault();
    if (!event.repeat) goBack();
  } else if (event.key === "ArrowRight" && empty) {
    event.preventDefault();
    if (!event.repeat) next();
  }
});

document.addEventListener("keydown", (event) => {
  if ($("task").hidden || event.target === $("entry")) return;
  if (event.key === "Escape") return closeSheet();
  if (!$("sheet").hidden) return;
  if (event.key === "ArrowLeft") goBack();
  if (event.key === "ArrowRight") next();
});

/* Bounding boxes (optional): drag on the image, then choose its object. ------ */

function sizeKey() {
  return `recognition.size.${username}`;
}

function loadSize() {
  try {
    const stored = localStorage.getItem(sizeKey());
    if (stored) $("size-range").value = stored;
  } catch { /* defaults to the largest size */ }
}

/* The slider sets a share of the space available; boxes follow as percentages. */
function fitOverlay() {
  const img = $("image");
  const overlay = $("boxes");
  const stage = $("stage");
  if (img.naturalWidth) {
    const room = Math.min(stage.clientWidth / img.naturalWidth, stage.clientHeight / img.naturalHeight);
    const scale = (room * Number($("size-range").value)) / 100;
    img.style.width = `${Math.round(img.naturalWidth * scale)}px`;
    img.style.height = `${Math.round(img.naturalHeight * scale)}px`;
  }
  overlay.style.left = `${img.offsetLeft}px`;
  overlay.style.top = `${img.offsetTop}px`;
  overlay.style.width = `${img.offsetWidth}px`;
  overlay.style.height = `${img.offsetHeight}px`;
}

function boxElement(box, className) {
  const img = $("image");
  const el = document.createElement("div");
  el.className = className;
  el.style.left = `${(100 * box.x) / img.naturalWidth}%`;
  el.style.top = `${(100 * box.y) / img.naturalHeight}%`;
  el.style.width = `${(100 * box.w) / img.naturalWidth}%`;
  el.style.height = `${(100 * box.h) / img.naturalHeight}%`;
  return el;
}

function renderBoxes() {
  const parts = boxes.map((box) => {
    const el = boxElement(box, "box");
    const tag = document.createElement("span");
    tag.textContent = box.label;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Remove box for ${box.label}`);
    remove.onclick = () => {
      boxes = boxes.filter((b) => b !== box);
      renderBoxes();
      renderObjects();
    };
    el.append(tag, remove);
    return el;
  });
  const shape = draft || pendingBox;
  if (shape) parts.push(boxElement(shape, "box draft"));
  $("boxes").replaceChildren(...parts);
}

function imagePoint(event) {
  const img = $("image");
  const rect = $("boxes").getBoundingClientRect();
  const clamp = (value) => Math.min(Math.max(value, 0), 1);
  return {
    x: Math.round(clamp((event.clientX - rect.left) / rect.width) * img.naturalWidth),
    y: Math.round(clamp((event.clientY - rect.top) / rect.height) * img.naturalHeight),
  };
}

function shapeFrom(a, b) {
  return { x: Math.min(a.x, b.x), y: Math.min(a.y, b.y), w: Math.abs(a.x - b.x), h: Math.abs(a.y - b.y) };
}

let dragStart = null;

$("boxes").addEventListener("pointerdown", (event) => {
  if (!shownAt || event.target.closest("button")) return;
  dragStart = imagePoint(event);
  $("boxes").setPointerCapture(event.pointerId);
  event.preventDefault();
});

$("boxes").addEventListener("pointermove", (event) => {
  if (!dragStart) return;
  draft = shapeFrom(dragStart, imagePoint(event));
  renderBoxes();
});

function endDrag() {
  if (!dragStart) return;
  const shape = draft;
  dragStart = null;
  draft = null;
  const scale = $("image").naturalWidth / $("boxes").clientWidth;
  if (shape && shape.w >= 10 * scale && shape.h >= 10 * scale) {
    addTyped();
    pendingBox = shape;
    if (picked.length === 1) mapBox(picked[0]);
    else openChooser();
  }
  renderBoxes();
  if (!touch) $("entry").focus();
}
$("boxes").addEventListener("pointerup", endDrag);
$("boxes").addEventListener("pointercancel", endDrag);

function mapBox(label) {
  boxes.push({ ...pendingBox, label });
  closeChooser();
  renderBoxes();
  renderObjects();
  if (!touch) $("entry").focus();
}

function openChooser() {
  const box = $("chooser");
  const title = document.createElement("span");
  title.textContent = picked.length ? "This box shows:" : "Type its object and press Enter";
  const options = picked.map((label, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = touch ? label : `${i + 1}. ${label}`;
    b.onclick = () => mapBox(label);
    return b;
  });
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "cancel";
  cancel.textContent = "Cancel";
  cancel.onclick = () => {
    closeChooser();
    renderBoxes();
  };
  box.replaceChildren(title, ...options, cancel);
  box.hidden = false;
}

function closeChooser() {
  pendingBox = null;
  $("chooser").hidden = true;
}

new ResizeObserver(fitOverlay).observe($("stage"));

$("size-range").addEventListener("input", (event) => {
  fitOverlay();
  try { localStorage.setItem(sizeKey(), event.target.value); } catch { /* this visit only */ }
});
$("size-range").addEventListener("change", () => { if (!touch) $("entry").focus(); });

/* Progress ---------------------------------------------------------------- */

async function loadMap() {
  try {
    const map = await api("/api/map");
    taskMap = { ids: map.ids, status: map.status.split("") };
  } catch { /* the plain counter still works */ }
}

function mark(id, status) {
  const i = taskMap.ids.indexOf(id);
  if (i >= 0) taskMap.status[i] = status;
  paintBar();
}

function moveToEnd(id) {
  const i = taskMap.ids.indexOf(id);
  if (i < 0) return;
  taskMap.ids.push(taskMap.ids.splice(i, 1)[0]);
  taskMap.status.splice(i, 1);
  taskMap.status.push("s");
  paintBar();
}

function color(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function paintBar() {
  const canvas = $("bar-canvas");
  const width = canvas.clientWidth;
  if (!width) return;
  const scale = window.devicePixelRatio || 1;
  canvas.width = width * scale;
  canvas.height = canvas.clientHeight * scale;
  const ctx = canvas.getContext("2d");
  const n = taskMap.status.length;
  ctx.fillStyle = color("--line");
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!n) return;
  const step = canvas.width / n;
  taskMap.status.forEach((s, i) => {
    if (s === "r") return;
    ctx.fillStyle = color(COLORS[s]);
    ctx.fillRect(Math.floor(i * step), 0, Math.max(1, Math.ceil(step)), canvas.height);
  });
}

function renderCounter() {
  const left = Math.max(total - done, 0);
  $("left").textContent = left.toLocaleString();
  let pace = paceMs;
  if (recent.length >= 5) pace = recent.slice().sort((a, b) => a - b)[recent.length >> 1];
  if (!pace) return void ($("eta").textContent = "");
  const minutes = Math.round((left * pace) / 60000);
  $("eta").textContent = minutes >= 60
    ? `${Math.floor(minutes / 60)}h ${minutes % 60}m`
    : `${Math.max(minutes, 1)}m`;
}

/* Sheets ------------------------------------------------------------------ */

function openSheet(title, part) {
  $("sheet-title").textContent = title;
  $("sheet-tasks").hidden = part !== "tasks";
  $("sheet-menu").hidden = part !== "menu";
  $("sheet").hidden = false;
}

function closeSheet() {
  $("sheet").hidden = true;
  if (!touch && !$("task").hidden) $("entry").focus();
}

$("sheet-close").addEventListener("click", closeSheet);
$("menu").addEventListener("click", () => openSheet("Menu", "menu"));
$("signout").addEventListener("click", async () => {
  await flush();
  await api("/api/logout", {});
  location.reload();
});

const CELL = 7;

function gridColumns(canvas) {
  return Math.max(1, Math.floor(canvas.clientWidth / CELL));
}

async function openTasks() {
  await flush();
  await loadMap();
  openSheet("All tasks", "tasks");
  const counts = { d: 0, u: 0, s: 0, r: 0 };
  taskMap.status.forEach((s) => { counts[s] += 1; });
  $("tally").replaceChildren(...[["d", "Done"], ["u", "Unidentified"], ["s", "Skipped"], ["r", "Remaining"]].map(([key, label]) => {
    const span = document.createElement("span");
    const dot = document.createElement("i");
    dot.style.background = color(COLORS[key]);
    span.append(dot, `${label} ${counts[key].toLocaleString()}`);
    return span;
  }));
  const canvas = $("grid");
  const columns = gridColumns(canvas);
  const rows = Math.ceil(taskMap.status.length / columns);
  const scale = window.devicePixelRatio || 1;
  canvas.style.height = `${rows * CELL}px`;
  canvas.width = columns * CELL * scale;
  canvas.height = rows * CELL * scale;
  const ctx = canvas.getContext("2d");
  ctx.scale(scale, scale);
  taskMap.status.forEach((s, i) => {
    ctx.fillStyle = color(COLORS[s]);
    ctx.fillRect((i % columns) * CELL, Math.floor(i / columns) * CELL, CELL - 1, CELL - 1);
  });
}

$("bar").addEventListener("click", openTasks);
$("grid").addEventListener("click", (event) => {
  const canvas = event.target;
  const box = canvas.getBoundingClientRect();
  const columns = gridColumns(canvas);
  const col = Math.floor((event.clientX - box.left) / CELL);
  const row = Math.floor((event.clientY - box.top) / CELL);
  const id = col < columns ? taskMap.ids[row * columns + col] : null;
  if (id) jumpTo(id);
});

window.addEventListener("resize", paintBar);

/* iOS keeps 100dvh when the keyboard opens; follow the visual viewport instead. */
if (window.visualViewport) {
  const fit = () => {
    document.documentElement.style.setProperty("--app-height", `${visualViewport.height}px`);
    window.scrollTo(0, 0);
  };
  visualViewport.addEventListener("resize", fit);
  fit();
}

start();
