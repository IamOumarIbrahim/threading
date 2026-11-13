"use strict";

/* Chaptered walkthrough of the task, animated with anime.js. */
window.tutorial = (() => {
  const $ = (id) => document.getElementById(id);
  const svgNS = "http://www.w3.org/2000/svg";
  const START_LEFT = 4812;
  const PACE = 1.6;        // stretches every chapter's timing
  const REPLAY_GAP = 2500; // pause before a chapter repeats

  let timers = [];
  let index = 0;
  let finished = false;
  let caret = null;
  let built = false;
  let left = START_LEFT;

  /* Stage helpers ---------------------------------------------------------- */

  const at = (ms, fn) => timers.push(setTimeout(fn, ms * PACE));

  function centre(el) {
    const frame = $("demo").getBoundingClientRect();
    const box = el.getBoundingClientRect();
    return { x: box.left - frame.left + box.width / 2, y: box.top - frame.top + box.height / 2 };
  }

  function pointTo(el, ms = 800) {
    const { x, y } = centre(el);
    anime({ targets: "#d-pointer", opacity: 1, left: x, top: y, duration: ms, easing: "easeInOutCubic" });
  }

  function press(el) {
    anime({ targets: el, scale: [1, 0.9, 1], duration: 260, easing: "easeInOutQuad" });
    anime({ targets: "#d-pointer", scale: [1, 0.7, 1], duration: 260, easing: "easeInOutQuad" });
  }

  function key(label) {
    const el = $("d-key");
    el.textContent = label;
    anime.remove(el);
    anime({ targets: el, opacity: [{ value: 1, duration: 100 }, { value: 1, duration: 900 }, { value: 0, duration: 250 }], easing: "easeOutQuad" });
  }

  function type(text) {
    const box = { n: 0 };
    $("d-text").textContent = "";
    anime({
      targets: box,
      n: text.length,
      round: 1,
      duration: Math.min(text.length * 110, 2000),
      easing: "linear",
      update: () => { $("d-text").textContent = text.slice(0, box.n); },
    });
  }

  function chip(text, flag) {
    const el = document.createElement("span");
    el.className = `d-chip${flag ? " flag" : ""}`;
    const name = document.createElement("span");
    name.textContent = text;
    const x = document.createElement("b");
    x.textContent = "×";
    el.append(name, x);
    $("d-list").append(el);
    anime({ targets: el, opacity: [0, 1], translateX: [-20, 0], duration: 320, easing: "easeOutBack" });
    return el;
  }

  function enterText() {
    key("Enter");
    const text = $("d-text").textContent;
    at(150, () => {
      chip(text);
      $("d-text").textContent = "";
    });
  }

  function nextImage(count = true) {
    anime({ targets: "#d-image", opacity: [1, 0.1, 1], duration: 600, easing: "easeInOutQuad" });
    at(280, () => {
      $("d-list").replaceChildren();
      $("d-frame").style.opacity = 0;
      if (count) tick();
    });
  }

  function tick() {
    left -= 1;
    $("d-left").textContent = left.toLocaleString();
    $("d-fill").style.width = `${6 + (START_LEFT - left) * 4}%`;
  }

  function drawFrame() {
    anime({ targets: "#d-frame", opacity: [0, 1], width: ["0%", "62%"], height: ["0%", "60%"], duration: 1280, easing: "easeOutQuad" });
  }

  function drawLines() {
    const svg = $("d-lines");
    const shape = [[40, 95, 62, 70], [62, 70, 105, 68], [105, 68, 122, 50], [122, 50, 140, 54], [140, 54, 146, 68],
      [146, 68, 128, 74], [128, 74, 120, 92], [120, 92, 122, 122], [100, 92, 102, 122], [66, 92, 62, 122],
      [46, 92, 44, 122], [40, 95, 22, 80], [62, 92, 120, 92], [158, 112, 172, 108], [172, 108, 176, 120],
      [176, 120, 162, 126], [162, 126, 158, 112], [10, 128, 192, 126]];
    shape.forEach(([x1, y1, x2, y2], i) => {
      for (let n = 0; n < 3; n += 1) {
        const line = document.createElementNS(svgNS, "line");
        const jitter = () => (n * 7.3 + i * 3.1) % 3 - 1.5;
        line.setAttribute("x1", x1 + jitter());
        line.setAttribute("y1", y1 - jitter());
        line.setAttribute("x2", x2 - jitter());
        line.setAttribute("y2", y2 + jitter());
        line.setAttribute("stroke", i > 12 && i < 17 ? "#b4502a" : "#3b3024");
        line.setAttribute("stroke-width", "1");
        svg.append(line);
      }
    });
  }

  function build() {
    if (built) return;
    built = true;
    $("demo").innerHTML = `
      <div class="d-top"><span id="d-help">Help</span><span><b id="d-left"></b> left</span><span id="d-menu">Menu</span></div>
      <div id="d-bar"><i id="d-fill"></i><i id="d-skip"></i></div>
      <div class="d-main">
        <div class="d-stage"><div id="d-image"><svg id="d-lines" viewBox="0 0 200 150"></svg><i id="d-frame"></i></div></div>
        <div class="d-side"><div class="d-title">Recognized objects</div><div id="d-list"></div></div>
      </div>
      <div class="d-entry"><div class="d-input"><span id="d-text"></span><i id="d-caret"></i></div>
        <div class="d-size">Size<span class="d-track"><i id="d-thumb"></i></span></div></div>
      <div class="d-actions"><span id="d-back">← Back</span><span id="d-unid">Unidentified</span><span id="d-next">Next →</span></div>
      <div id="d-chooser">This box shows: <span id="d-option">dog</span></div>
      <div id="d-sheet"></div>
      <div id="d-closed">Page closed. Sign in to continue.</div>
      <div id="d-key"></div>
      <i id="d-pointer"></i>`;
    drawLines();
  }

  function reset() {
    anime.remove("#d-pointer, #d-image, #d-frame, #d-thumb, #d-key, #d-closed");
    $("d-list").replaceChildren();
    $("d-text").textContent = "";
    for (const id of ["d-key", "d-frame", "d-pointer", "d-closed"]) $(id).style.opacity = 0;
    for (const id of ["d-chooser", "d-sheet"]) $(id).hidden = true;
    $("d-frame").style.width = 0;
    $("d-frame").style.height = 0;
    $("d-image").style.transform = "";
    $("d-image").style.opacity = 1;
    $("d-pointer").style.transform = "";
    $("d-thumb").style.left = "100%";
    $("d-skip").style.opacity = 0;
    left = START_LEFT;
    $("d-left").textContent = left.toLocaleString();
    $("d-fill").style.width = "6%";
    const { x, y } = centre($("d-text"));
    $("d-pointer").style.left = `${x}px`;
    $("d-pointer").style.top = `${y}px`;
  }

  function sheet(kind) {
    const el = $("d-sheet");
    el.hidden = false;
    if (kind === "tasks") {
      const cells = Array.from({ length: 60 }, (_, i) => {
        const type = i < 10 ? "done" : i < 13 ? "unid" : i >= 57 ? "skip" : "";
        return `<i class="${type}"></i>`;
      }).join("");
      el.innerHTML = `<b>All tasks</b><div class="d-grid">${cells}</div>
        <div class="d-legend"><span class="done">Done</span><span class="unid">Unidentified</span><span class="skip">Skipped</span><span>Remaining</span></div>`;
    } else {
      el.innerHTML = "<b>Menu</b><div class=\"d-opt\">Light mode</div><div class=\"d-opt\">Sign out</div>";
    }
    anime({ targets: el, translateY: [40, 0], opacity: [0, 1], duration: 300, easing: "easeOutQuad" });
  }

  /* Chapters --------------------------------------------------------------- */

  const CHAPTERS = [
    { title: "Type what you see", computer: "Type, then press Enter.", phone: "Type, then tap Send on the keyboard.", ms: 3400, run() {
      at(300, () => type("dog"));
      at(1300, enterText);
    } },
    { title: "Any format works", computer: "A word, a list or a full sentence. Enter after each.", phone: "A word, a list or a full sentence. Send after each.", ms: 5600, run() {
      at(200, () => type("cat, ball"));
      at(1300, enterText);
      at(2200, () => type("a dog chasing a ball"));
      at(3900, enterText);
    } },
    { title: "Remove a mistake", computer: "Click × next to it, or press Backspace on an empty box.", phone: "Tap × next to it.", ms: 3800, run() {
      const first = chip("dgo");
      chip("dog");
      at(500, () => pointTo(first.lastChild));
      at(1300, () => press(first.lastChild));
      at(1500, () => first.remove());
    } },
    { title: "Go to the next image", computer: "Press Enter on an empty box, press →, or click Next.", phone: "Tap Next.", ms: 3800, run() {
      chip("dog");
      at(500, () => pointTo($("d-next")));
      at(1300, () => press($("d-next")));
      at(1500, nextImage);
    } },
    { title: "Recognise nothing?", computer: "Press Enter on an empty box, or click Unidentified.", phone: "Tap Unidentified.", ms: 4000, run() {
      at(500, () => pointTo($("d-unid")));
      at(1300, () => press($("d-unid")));
      at(1500, () => chip("Unidentified", true));
      at(2500, nextImage);
    } },
    { title: "Go back and change an answer", computer: "Press ← or click Back. Press Enter when done.", phone: "Tap Back, change it, then tap Next.", ms: 5000, run() {
      at(400, () => pointTo($("d-back")));
      at(1100, () => { key("←"); press($("d-back")); });
      at(1300, () => { nextImage(false); at(320, () => chip("dog")); });
      at(2200, () => type("ball"));
      at(3100, enterText);
    } },
    { title: "Skip an image for now", computer: "Press → with nothing entered.", phone: "Tap Next with nothing entered.", note: "Skipped images come back at the end.", ms: 4400, run() {
      at(400, () => pointTo($("d-next")));
      at(1200, () => { key("→"); press($("d-next")); });
      at(1400, () => nextImage(false));
      at(2000, () => anime({ targets: "#d-skip", opacity: [0, 1], scaleX: [3, 1], duration: 500, easing: "easeOutQuad" }));
    } },
    { title: "Draw boxes (optional)", computer: "Drag on the image, then pick the object it shows.", phone: "Drag with your finger, then tap the object.", note: "× on a box removes it. You never have to draw one.", ms: 6000, run() {
      chip("dog");
      chip("ball");
      const image = $("d-image").getBoundingClientRect();
      const frame = $("demo").getBoundingClientRect();
      at(300, () => {
        $("d-pointer").style.opacity = 1;
        anime({ targets: "#d-pointer", left: image.left - frame.left + image.width * 0.08, top: image.top - frame.top + image.height * 0.25, duration: 800, easing: "easeInOutCubic" });
      });
      at(900, () => {
        drawFrame();
        anime({ targets: "#d-pointer", left: image.left - frame.left + image.width * 0.7, top: image.top - frame.top + image.height * 0.85, duration: 1280, easing: "easeOutQuad" });
      });
      at(1900, () => {
        $("d-chooser").hidden = false;
        anime({ targets: "#d-chooser", opacity: [0, 1], duration: 220, easing: "linear" });
      });
      at(2500, () => pointTo($("d-option")));
      at(3200, () => press($("d-option")));
      at(3400, () => { $("d-chooser").hidden = true; });
    } },
    { title: "Resize the image", computer: "Drag the Size slider.", phone: "Drag the Size slider.", note: "Your size is remembered on this device.", ms: 4200, run() {
      at(300, () => pointTo($("d-thumb")));
      at(1000, () => {
        anime({ targets: "#d-thumb", left: ["100%", "20%", "75%"], duration: 3500, easing: "easeInOutQuad" });
        anime({ targets: "#d-pointer", translateX: [0, -48, -16], duration: 3500, easing: "easeInOutQuad" });
        anime({ targets: "#d-image", scale: [1, 0.5, 0.85], duration: 3500, easing: "easeInOutQuad" });
      });
    } },
    { title: "Track your tasks", computer: "The top bar shows done, unidentified, skipped and remaining. Click it to see every task.", phone: "The top bar shows done, unidentified, skipped and remaining. Tap it to see every task.", note: "Open any square to go to that image.", ms: 5000, run() {
      at(400, () => pointTo($("d-bar")));
      at(1200, () => press($("d-bar")));
      at(1400, () => sheet("tasks"));
    } },
    { title: "Stop and return any time", computer: "Answers save by themselves. There is no save button.", phone: "Answers save by themselves. There is no save button.", note: "Sign in again to continue. The image left open returns at the end.", ms: 5000, run() {
      chip("dog");
      at(800, () => anime({ targets: "#d-closed", opacity: [0, 1], duration: 400, easing: "linear" }));
      at(3000, () => anime({ targets: "#d-closed", opacity: 0, duration: 400, easing: "linear" }));
    } },
    { title: "Help and appearance", computer: "Menu has Light or Dark mode and Sign out. Help reopens this guide.", phone: "Menu has Light or Dark mode and Sign out. Help reopens this guide.", ms: 5000, run() {
      at(400, () => pointTo($("d-menu")));
      at(1100, () => press($("d-menu")));
      at(1300, () => sheet("menu"));
      at(2900, () => { $("d-sheet").hidden = true; });
      at(3100, () => pointTo($("d-help")));
      at(3800, () => press($("d-help")));
    } },
  ];

  /* Player: moves only when the person presses Next. ------------------------ */

  function clearTimers() {
    timers.forEach(clearTimeout);
    timers = [];
  }

  function caption(chapter) {
    $("t-count").textContent = `${index + 1} of ${CHAPTERS.length}`;
    $("t-title").textContent = chapter.title;
    $("t-computer").textContent = chapter.computer;
    $("t-phone").textContent = chapter.phone;
    $("t-note").textContent = chapter.note || "";
  }

  function go(n) {
    clearTimers();
    index = Math.min(Math.max(n, 0), CHAPTERS.length - 1);
    const chapter = CHAPTERS[index];
    caption(chapter);
    $("t-prev").disabled = index === 0;
    $("t-next").disabled = index === CHAPTERS.length - 1;
    if (index === CHAPTERS.length - 1) finished = true;
    $("intro-start").hidden = !finished;
    reset();
    if (!window.anime) return;
    chapter.run();
    at(chapter.ms + REPLAY_GAP, () => go(index)); // repeats until they move on
  }

  function steps() {
    const list = $("t-steps");
    if (list.childElementCount) return;
    list.append(...CHAPTERS.map((c) => {
      const li = document.createElement("li");
      const title = document.createElement("strong");
      title.textContent = `${c.title}. `;
      const phone = c.computer === c.phone ? "" : ` Phone: ${c.phone}`;
      li.append(title, `${c.computer}${phone}${c.note ? ` ${c.note}` : ""}`);
      return li;
    }));
  }

  /* firstTime: Start only appears after the last chapter. */
  function play(firstTime) {
    build();
    steps();
    finished = !firstTime;
    $("intro-start").textContent = firstTime ? "Start" : "Back to task";
    if (window.anime && !caret) {
      caret = anime({ targets: "#d-caret", opacity: [1, 0], duration: 500, direction: "alternate", loop: true, easing: "steps(1)" });
    }
    go(0);
  }

  function stop() {
    clearTimers();
    if (caret) caret.pause();
    caret = null;
  }

  $("t-prev").addEventListener("click", () => go(index - 1));
  $("t-next").addEventListener("click", () => go(index + 1));
  $("t-replay").addEventListener("click", () => go(index));

  return { play, stop, go, chapters: CHAPTERS.length };
})();
