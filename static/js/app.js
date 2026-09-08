/* Multimodal Sentiment Analysis — front end */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const FACES = { positive: "😊", neutral: "😐", negative: "☹️" };
  const HISTORY_KEY = "msa.history.v1";
  const MAX_HISTORY = 25;

  const store = {
    get(key, fallback) {
      try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; }
      catch (_) { return fallback; }
    },
    set(key, value) {
      try { localStorage.setItem(key, JSON.stringify(value)); } catch (_) { /* quota / private mode */ }
    },
  };

  // ---------------------------------------------------------------- theme
  const root = document.documentElement;
  try {
    const saved = localStorage.getItem("theme");
    if (saved) root.setAttribute("data-theme", saved);
  } catch (_) {}

  $("theme-toggle").addEventListener("click", () => {
    const dark = root.getAttribute("data-theme") === "dark" ||
      (!root.getAttribute("data-theme") && matchMedia("(prefers-color-scheme: dark)").matches);
    const next = dark ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("theme", next); } catch (_) {}
  });

  // ---------------------------------------------------------------- toasts
  function toast(message, bad) {
    const el = document.createElement("div");
    el.className = "toast" + (bad ? " bad" : "");
    el.textContent = message;
    $("toasts").appendChild(el);
    setTimeout(() => {
      el.style.opacity = "0";
      el.style.transition = "opacity .3s";
      setTimeout(() => el.remove(), 300);
    }, 4200);
  }

  // ---------------------------------------------------------------- tabs
  const tabs = [...document.querySelectorAll(".tab")];
  function showTab(name) {
    tabs.forEach((t) => t.setAttribute("aria-selected", String(t.dataset.panel === name)));
    document.querySelectorAll(".panel").forEach((p) =>
      p.classList.toggle("active", p.id === "panel-" + name));
    if (name === "history") renderHistory();
  }
  tabs.forEach((tab) => tab.addEventListener("click", () => showTab(tab.dataset.panel)));
  $("history-btn").addEventListener("click", () => showTab("history"));

  document.addEventListener("keydown", (e) => {
    const typing = /^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName);
    if (e.key >= "1" && e.key <= "6" && !typing && !e.ctrlKey && !e.metaKey) {
      const tab = tabs[Number(e.key) - 1];
      if (tab) showTab(tab.dataset.panel);
    }
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      const active = document.querySelector(".panel.active");
      const go = active && active.querySelector("button.primary:not(:disabled)");
      if (go) go.click();
    }
  });

  // ---------------------------------------------------------------- helpers
  const pct = (v) => (v * 100).toFixed(1) + "%";
  const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  /** Circular confidence ring. */
  function ringSvg(value, label) {
    const r = 26, c = 2 * Math.PI * r, filled = c * Math.min(1, Math.max(0, value));
    return `<svg class="ring" width="68" height="68" viewBox="0 0 68 68" aria-hidden="true">
      <circle cx="34" cy="34" r="${r}" fill="none" stroke="var(--surface-3)" stroke-width="6"/>
      <circle cx="34" cy="34" r="${r}" fill="none" stroke="currentColor" stroke-width="6"
        stroke-linecap="round" stroke-dasharray="${filled} ${c}" transform="rotate(-90 34 34)"/>
      <text x="34" y="31" text-anchor="middle" font-size="13" font-weight="700"
        fill="var(--text)">${Math.round(value * 100)}%</text>
      <text x="34" y="44" text-anchor="middle" font-size="7.5" fill="var(--text-faint)">${label}</text>
    </svg>`;
  }

  function verdictBlock(r, title) {
    const label = r.label || "unknown";
    return `<div class="verdict ${label}">
      <div class="face">${FACES[label] || "❓"}</div>
      <div class="who">
        <div class="label">${label}</div>
        <div class="sub">${escapeHtml(title)} · ${pct(r.confidence)} confidence · analysed in
          ${r.elapsed_ms > 1500 ? (r.elapsed_ms / 1000).toFixed(1) + " s" : r.elapsed_ms + " ms"}</div>
      </div>
      ${ringSvg(r.certainty, "certainty")}
    </div>`;
  }

  function barsBlock(dist) {
    return `<div class="bars">` + ["positive", "neutral", "negative"].map((k) => `
      <div class="bar-row">
        <span>${k}</span>
        <span class="bar-track"><span class="bar-fill ${k}" data-w="${(dist[k] || 0) * 100}"></span></span>
        <span class="bar-val">${pct(dist[k] || 0)}</span>
      </div>`).join("") + `</div>`;
  }

  function meterBlock(valence) {
    return `<div class="meter">
      <div class="meter-track"><div class="meter-pin" data-left="${((valence + 1) / 2) * 100}"></div></div>
      <div class="meter-scale"><span>−1 negative</span><span>0</span><span>+1 positive</span></div>
      <div class="meter-value">valence ${valence.toFixed(3)}</div>
    </div>`;
  }

  function statsBlock(pairs) {
    const shown = pairs.filter((p) => p[1] !== undefined && p[1] !== null && p[1] !== "");
    if (!shown.length) return "";
    return `<div class="stats">` +
      shown.map(([k, v]) => `<span class="stat">${k} <b>${escapeHtml(v)}</b></span>`).join("") + `</div>`;
  }

  function warnBlock(warnings) {
    if (!warnings || !warnings.length) return "";
    return `<div class="alert warn"><strong>Worth knowing</strong><ul>` +
      warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("") + `</ul></div>`;
  }

  const errorBlock = (msg) =>
    `<div class="alert error"><strong>Could not analyse this</strong>${escapeHtml(msg)}</div>`;

  /** Emotion tiles, sorted strongest first. */
  function emotionsBlock(emotions, asPercent) {
    const entries = Object.entries(emotions).sort((a, b) => b[1] - a[1]);
    if (!entries.length) return "";
    const max = entries[0][1] || 1;
    return `<div class="emo-grid">` + entries.map(([name, v]) => `
      <div class="emo">
        <div class="n">${escapeHtml(name)}</div>
        <div class="v">${asPercent ? v.toFixed(1) + "%" : pct(v)}</div>
        <div class="t"><i style="width:${(v / max) * 100}%"></i></div>
      </div>`).join("") + `</div>`;
  }

  /** Valence over time as an area chart. */
  function timelineChart(timeline) {
    if (!timeline || timeline.length < 2) return "";
    const W = 720, H = 130, pad = 4;
    const maxT = timeline[timeline.length - 1].t || timeline.length;
    const x = (t) => pad + (t / (maxT || 1)) * (W - pad * 2);
    const y = (v) => H / 2 - v * (H / 2 - pad);

    const pts = timeline.map((p) => `${x(p.t).toFixed(1)},${y(p.valence).toFixed(1)}`);
    const area = `M ${x(0).toFixed(1)},${(H / 2).toFixed(1)} L ` + pts.join(" L ") +
      ` L ${x(maxT).toFixed(1)},${(H / 2).toFixed(1)} Z`;

    const dots = timeline.map((p) => {
      const cls = p.label || "neutral";
      const colour = cls === "positive" ? "var(--positive)"
        : cls === "negative" ? "var(--negative)" : "var(--neutral)";
      return `<circle cx="${x(p.t).toFixed(1)}" cy="${y(p.valence).toFixed(1)}" r="2.6"
        fill="${colour}"><title>${p.t}s · ${escapeHtml(p.emotion || cls)} · valence ${p.valence}</title></circle>`;
    }).join("");

    return `<div class="chart-wrap">
      <div class="chart-title"><span>Sentiment over time</span><span>${maxT.toFixed(0)}s</span></div>
      <svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="height:130px">
        <defs>
          <linearGradient id="tlg" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="var(--positive)" stop-opacity=".35"/>
            <stop offset="50%" stop-color="var(--neutral)" stop-opacity=".08"/>
            <stop offset="100%" stop-color="var(--negative)" stop-opacity=".35"/>
          </linearGradient>
        </defs>
        <line x1="0" y1="${H / 2}" x2="${W}" y2="${H / 2}" stroke="var(--border)" stroke-width="1"/>
        <path d="${area}" fill="url(#tlg)"/>
        <polyline points="${pts.join(" ")}" fill="none" stroke="var(--accent)" stroke-width="1.8"
          stroke-linejoin="round" stroke-linecap="round"/>
        ${dots}
      </svg>
    </div>`;
  }

  function tableBlock(rows, headers) {
    if (!rows || !rows.length) return "";
    return `<div class="scroll-x"><table class="data"><thead><tr>` +
      headers.map((h) => `<th>${h}</th>`).join("") + `</tr></thead><tbody>` +
      rows.map((r) => `<tr>` + r.map((c) => `<td>${c}</td>`).join("") + `</tr>`).join("") +
      `</tbody></table></div>`;
  }

  function detailBlock(r) {
    const d = r.detail || {};
    let html = "";

    if (r.modality === "text") {
      html += statsBlock([["words", d.words], ["characters", d.characters], ["windows", d.windows]]);
      if (d.per_window && d.per_window.length) {
        html += `<details class="evidence"><summary>Per-window breakdown</summary><div>` +
          tableBlock(d.per_window.map((w) => [escapeHtml(w.excerpt), w.label, pct(w.confidence), w.valence]),
            ["Excerpt", "Label", "Confidence", "Valence"]) + `</div></details>`;
      }
    }

    if (r.modality === "image") {
      html += statsBlock([["faces", d.face_count],
        ["size", d.width && d.height ? `${d.width}×${d.height}` : null]]);
      if (d.faces && d.faces.length) {
        html += emotionsBlock(d.faces[0].emotions, false);
        if (d.faces.length > 1) {
          html += `<details class="evidence"><summary>All ${d.faces.length} faces</summary><div>` +
            tableBlock(d.faces.map((f, i) => ["Face " + (i + 1), f.dominant_emotion, f.label,
              pct(f.confidence), `${f.region.w}×${f.region.h}`, f.face_confidence]),
              ["#", "Emotion", "Sentiment", "Confidence", "Size", "Detection"]) + `</div></details>`;
        }
      }
    }

    if (r.modality === "audio") {
      html += statsBlock([["duration", d.duration_seconds ? d.duration_seconds + "s" : null],
        ["windows", d.windows], ["dominant tone", d.dominant_emotion]]);
      html += timelineChart(d.timeline);
      if (d.transcript) {
        html += `<details class="evidence"><summary>Transcript</summary><div>
          <p style="font-size:.87rem">${escapeHtml(d.transcript)}</p></div></details>`;
      }
    }

    if (r.modality === "video") {
      html += statsBlock([
        ["duration", d.duration_seconds ? d.duration_seconds + "s" : null],
        ["frames", d.frames_sampled], ["with faces", d.frames_with_faces],
        ["detection rate", d.face_detection_rate !== undefined ? pct(d.face_detection_rate) : null],
        ["dominant", d.dominant_emotion], ["fusion", d.fusion],
      ]);
      html += timelineChart(d.timeline);
      if (d.emotion_counts) {
        const total = Object.values(d.emotion_counts).reduce((a, b) => a + b, 0) || 1;
        const shares = {};
        Object.entries(d.emotion_counts).forEach(([k, v]) => { shares[k] = v / total; });
        html += emotionsBlock(shares, false);
      }
      if (d.audio) {
        html += `<details class="evidence"><summary>Audio track</summary><div>` +
          statsBlock([["label", d.audio.label], ["confidence", pct(d.audio.confidence)],
            ["tone", d.audio.dominant_emotion]]) + `</div></details>`;
      }
    }

    if (r.modality === "fusion" && d.per_modality) {
      html += statsBlock([["modalities", (d.modalities || []).join(", ")],
        ["agreement", d.agreement ? "yes" : "no"]]);
      html += tableBlock(Object.entries(d.per_modality).map(([m, v]) =>
        [m, v.label, pct(v.confidence), v.valence]), ["Modality", "Label", "Confidence", "Valence"]);
    }

    html += `<details class="evidence"><summary>Raw JSON</summary><div>
      <pre class="json">${escapeHtml(JSON.stringify(r, null, 2))}</pre></div></details>`;
    return html;
  }

  /** Size the bars and valence pin once they are in the DOM.
   *
   * A timer is used rather than requestAnimationFrame because rAF is paused in
   * background tabs: if the user switches away while a long analysis runs, the
   * callback would never fire and every bar would render empty.
   */
  function animate(el) {
    setTimeout(() => {
      el.querySelectorAll(".bar-fill").forEach((b) => {
        const w = Number(b.dataset.w);
        b.style.width = (Number.isFinite(w) ? w : 0) + "%";
      });
      el.querySelectorAll(".meter-pin").forEach((p) => {
        const l = Number(p.dataset.left);
        p.style.left = (Number.isFinite(l) ? l : 50) + "%";
      });
    }, 30);
  }

  function render(targetId, r, title, opts) {
    const el = $(targetId);
    el.classList.add("show");
    if (!r || r.ok === false) {
      el.innerHTML = errorBlock((r && r.error) || "Analysis failed.");
      toast((r && r.error) ? r.error.slice(0, 90) : "Analysis failed", true);
      return;
    }
    el.innerHTML = verdictBlock(r, title) + barsBlock(r.distribution) +
      meterBlock(r.valence) + warnBlock(r.warnings) + detailBlock(r);
    animate(el);
    if (!opts || opts.remember !== false) addHistory(r, title);
  }

  function busy(button, on, label) {
    button.disabled = on;
    if (on) {
      button.dataset.text = button.dataset.text || button.textContent;
      button.innerHTML = `<span class="spinner"></span>${label || "Working…"}`;
      $("badge-models").textContent = "models running";
    } else {
      button.textContent = button.dataset.text || button.textContent;
      $("badge-models").textContent = "models ready";
    }
  }

  async function postJSON(url, body) {
    const res = await fetch(url, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    return res.json();
  }

  async function postFile(url, file) {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(url, { method: "POST", body: fd });
    if (res.status === 413) return { ok: false, error: "That file is larger than the upload limit." };
    return res.json();
  }

  // ---------------------------------------------------------------- history
  function addHistory(result, title) {
    const items = store.get(HISTORY_KEY, []);
    items.unshift({
      at: Date.now(), title,
      label: result.label, confidence: result.confidence, modality: result.modality,
      result,
    });
    store.set(HISTORY_KEY, items.slice(0, MAX_HISTORY));
  }

  function renderHistory() {
    const items = store.get(HISTORY_KEY, []);
    const list = $("history-list");
    if (!items.length) {
      list.innerHTML = `<div class="empty">Nothing yet — run an analysis and it will appear here.</div>`;
      return;
    }
    list.innerHTML = items.map((item, i) => {
      const when = new Date(item.at).toLocaleString();
      return `<div class="hist" data-i="${i}">
        <span class="face">${FACES[item.label] || "❓"}</span>
        <span class="what">${escapeHtml(item.title || item.modality)}</span>
        <span class="tagl ${item.label}">${item.label}</span>
        <span class="when">${escapeHtml(when)}</span>
      </div>`;
    }).join("");

    list.querySelectorAll(".hist").forEach((row) => {
      row.addEventListener("click", () => {
        const item = store.get(HISTORY_KEY, [])[Number(row.dataset.i)];
        if (item) render("history-result", item.result, item.title, { remember: false });
      });
    });
  }

  $("history-clear").addEventListener("click", () => {
    store.set(HISTORY_KEY, []);
    $("history-result").classList.remove("show");
    renderHistory();
    toast("History cleared");
  });

  // ---------------------------------------------------------------- text
  const SAMPLES = [
    ["Mixed review", "The battery life is outstanding and the screen is gorgeous. Setup took twenty minutes. Support, however, was slow and unhelpful, which soured an otherwise excellent experience."],
    ["Clearly positive", "Genuinely the best purchase I have made all year — it exceeded every expectation."],
    ["Clearly negative", "Arrived broken, the refund was refused, and nobody replied for a week. Avoid."],
    ["Factual / neutral", "The meeting is scheduled for 3pm in room B. The agenda was circulated on Monday."],
  ];

  $("text-samples").innerHTML = SAMPLES.map((s, i) =>
    `<button class="chip" data-i="${i}">${escapeHtml(s[0])}</button>`).join("");
  $("text-samples").addEventListener("click", (e) => {
    const btn = e.target.closest(".chip");
    if (!btn) return;
    textInput.value = SAMPLES[Number(btn.dataset.i)][1];
    textInput.dispatchEvent(new Event("input"));
  });

  const textInput = $("text-input");
  textInput.addEventListener("input", () => {
    const n = textInput.value.trim() ? textInput.value.trim().split(/\s+/).length : 0;
    $("text-count").textContent = n + (n === 1 ? " word" : " words");
  });

  $("text-clear").addEventListener("click", () => {
    textInput.value = "";
    textInput.dispatchEvent(new Event("input"));
    $("text-result").classList.remove("show");
  });

  $("text-go").addEventListener("click", async () => {
    const btn = $("text-go");
    const text = textInput.value.trim();
    if (!text) { render("text-result", { ok: false, error: "Enter some text first." }); return; }
    busy(btn, true, "Analysing…");
    try {
      const r = await postJSON("/api/analyze/text", { text });
      render("text-result", r, text.slice(0, 60) + (text.length > 60 ? "…" : ""));
    } catch (e) {
      render("text-result", { ok: false, error: String(e) });
    } finally { busy(btn, false); }
  });

  // ---------------------------------------------------------------- uploads
  function wireDrop(dropId, inputId, previewId, buttonId, kind) {
    const drop = $(dropId), input = $(inputId), preview = $(previewId), button = $(buttonId);
    let chosen = null;

    function show(file) {
      chosen = file;
      button.disabled = !file;
      if (!file) { preview.classList.remove("show"); preview.innerHTML = ""; return; }
      const url = URL.createObjectURL(file);
      const tag = kind === "image" ? `<img src="${url}" alt="preview">`
        : kind === "audio" ? `<audio controls src="${url}"></audio>`
        : `<video controls src="${url}"></video>`;
      preview.innerHTML = tag + `<div class="file-meta">
        <span>${escapeHtml(file.name)}</span>
        <span>· ${(file.size / 1048576).toFixed(2)} MB</span>
        <button class="ghost clear" type="button">Remove</button></div>`;
      preview.classList.add("show");
      preview.querySelector(".clear").addEventListener("click", (e) => {
        e.preventDefault();
        input.value = "";
        show(null);
      });
    }

    input.addEventListener("change", () => show(input.files[0] || null));
    drop.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
    });
    ["dragenter", "dragover"].forEach((ev) =>
      drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) =>
      drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", (e) => {
      const file = e.dataTransfer.files[0];
      if (file) { input.files = e.dataTransfer.files; show(file); }
    });

    return () => chosen;
  }

  const getImage = wireDrop("image-drop", "image-file", "image-preview", "image-go", "image");
  const getAudio = wireDrop("audio-drop", "audio-file", "audio-preview", "audio-go", "audio");
  const getVideo = wireDrop("video-drop", "video-file", "video-preview", "video-go", "video");

  async function runFile(buttonId, resultId, endpoint, getFile, kind) {
    const btn = $(buttonId), file = getFile();
    if (!file) return;
    busy(btn, true, "Analysing…");
    try {
      render(resultId, await postFile(endpoint, file), file.name);
    } catch (e) {
      render(resultId, { ok: false, error: String(e) });
    } finally { busy(btn, false); }
  }

  $("image-go").addEventListener("click", () =>
    runFile("image-go", "image-result", "/api/analyze/image", getImage, "image"));
  $("audio-go").addEventListener("click", () =>
    runFile("audio-go", "audio-result", "/api/analyze/audio", getAudio, "audio"));

  // ---------------------------------------------------------------- video jobs
  async function runVideoJob(file, prefix) {
    const wrap = $(prefix + "-progress"), fill = $(prefix + "-progress-fill");
    const msg = $(prefix + "-progress-msg"), pctEl = $(prefix + "-progress-pct");
    wrap.classList.add("show");
    fill.style.width = "3%";
    msg.textContent = "Uploading…";
    pctEl.textContent = "";

    const started = await postFile("/api/analyze/video", file);
    if (!started.ok) { wrap.classList.remove("show"); return started; }

    for (;;) {
      await new Promise((r) => setTimeout(r, 900));
      let job;
      try {
        job = await (await fetch("/api/jobs/" + started.job_id)).json();
      } catch (e) {
        wrap.classList.remove("show");
        return { ok: false, error: "Lost contact with the server: " + e };
      }
      if (job.ok === false) { wrap.classList.remove("show"); return job; }
      const p = Math.max(3, (job.progress || 0) * 100);
      fill.style.width = p + "%";
      msg.textContent = job.message || "Working…";
      pctEl.textContent = Math.round(p) + "%";
      if (job.state === "done") { wrap.classList.remove("show"); return job.result; }
      if (job.state === "error") {
        wrap.classList.remove("show");
        return { ok: false, error: job.message || "Video analysis failed." };
      }
    }
  }

  $("video-go").addEventListener("click", async () => {
    const btn = $("video-go"), file = getVideo();
    if (!file) return;
    busy(btn, true, "Analysing…");
    try {
      render("video-result", await runVideoJob(file, "video"), file.name);
    } catch (e) {
      render("video-result", { ok: false, error: String(e) });
    } finally { busy(btn, false); }
  });

  // ---------------------------------------------------------------- combined
  const comboFiles = { image: null, audio: null, video: null };

  ["image", "audio", "video"].forEach((kind) => {
    $("combo-pick-" + kind).addEventListener("click", () => $("combo-" + kind).click());
    $("combo-" + kind).addEventListener("change", (e) => {
      comboFiles[kind] = e.target.files[0] || null;
      const names = Object.entries(comboFiles).filter(([, f]) => f).map(([k, f]) => `${k}: ${f.name}`);
      $("combo-files").textContent = names.length ? names.join(" · ") : "nothing attached";
    });
  });

  $("combo-go").addEventListener("click", async () => {
    const btn = $("combo-go"), text = $("combo-text").value.trim();
    if (!text && !comboFiles.image && !comboFiles.audio && !comboFiles.video) {
      render("combo-result", { ok: false, error: "Add text or at least one file." });
      return;
    }

    busy(btn, true, "Analysing…");
    const wrap = $("combo-progress"), fill = $("combo-progress-fill");
    const msg = $("combo-progress-msg"), pctEl = $("combo-progress-pct");
    wrap.classList.add("show");

    const steps = [];
    if (text) steps.push(["text", () => postJSON("/api/analyze/text", { text })]);
    if (comboFiles.image) steps.push(["image", () => postFile("/api/analyze/image", comboFiles.image)]);
    if (comboFiles.audio) steps.push(["audio", () => postFile("/api/analyze/audio", comboFiles.audio)]);
    if (comboFiles.video) steps.push(["video", () => runVideoJob(comboFiles.video, "combo")]);

    const results = {};
    try {
      for (let i = 0; i < steps.length; i++) {
        const [name, run] = steps[i];
        msg.textContent = `Analysing ${name}… (${i + 1} of ${steps.length})`;
        pctEl.textContent = Math.round((i / steps.length) * 100) + "%";
        fill.style.width = ((i / steps.length) * 100) + "%";
        try { results[name] = await run(); }
        catch (e) { results[name] = { ok: false, error: String(e) }; }
      }

      wrap.classList.add("show");
      fill.style.width = "100%";
      pctEl.textContent = "100%";
      msg.textContent = "Fusing signals…";

      const usable = {};
      Object.entries(results).forEach(([k, v]) => { if (v && v.ok) usable[k] = v; });

      const el = $("combo-result");
      el.classList.add("show");

      if (!Object.keys(usable).length) {
        el.innerHTML = errorBlock("No input produced a usable result.") + failures(results);
        return;
      }

      const fused = await postJSON("/api/fuse", { results: usable });
      if (fused.ok) {
        el.innerHTML = verdictBlock(fused, "combined") + barsBlock(fused.distribution) +
          meterBlock(fused.valence) + warnBlock(fused.warnings) + detailBlock(fused) + failures(results);
        animate(el);
        addHistory(fused, "Combined: " + Object.keys(usable).join(" + "));
      } else {
        el.innerHTML = errorBlock(fused.error || "Fusion failed.") + failures(results);
      }
    } finally {
      wrap.classList.remove("show");
      busy(btn, false);
    }
  });

  function failures(results) {
    const failed = Object.entries(results).filter(([, v]) => !v || !v.ok);
    if (!failed.length) return "";
    return `<div class="alert warn" style="margin-top:14px">
      <strong>Some inputs could not be analysed</strong><ul>` +
      failed.map(([k, v]) => `<li>${k}: ${escapeHtml((v && v.error) || "failed")}</li>`).join("") +
      `</ul></div>`;
  }

  // ---------------------------------------------------------------- boot
  fetch("/api/health").then((r) => r.json()).then((h) => {
    $("badge-models").textContent = h.models_loaded && h.models_loaded.length
      ? `models ready (${h.models_loaded.join(", ")})` : "models load on first use";
  }).catch(() => { $("badge-models").textContent = "server unreachable"; });

  renderHistory();
})();
