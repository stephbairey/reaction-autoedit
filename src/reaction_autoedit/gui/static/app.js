/* Reaction AutoEdit GUI — vanilla JS, hash routing, polling. One file on purpose (no build step). */
"use strict";
const $ = (s, el = document) => el.querySelector(s);
const view = $("#view");
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmtT = t => { t = Math.round(t); const h = t/3600|0, m = (t%3600)/60|0, s = t%60; return (h? h+":"+String(m).padStart(2,"0") : m) + ":" + String(s).padStart(2,"0"); };

async function api(path, opts = {}) {
  const r = await fetch("/api" + path, { headers: {"Content-Type":"application/json"}, ...opts });
  if (!r.ok) { let d; try { d = (await r.json()).detail; } catch { d = r.statusText; } throw new Error(d); }
  return r.json();
}
function toast(msg, err = false) {
  const d = document.createElement("div");
  d.textContent = msg; if (err) d.className = "err";
  $("#toast").appendChild(d); setTimeout(() => d.remove(), err ? 9000 : 4500);
}

/* ---------- system badge ---------- */
async function sysBadge() {
  try {
    const s = await api("/system");
    $("#sys").innerHTML = `enc <b>${esc(s.encoder)}</b> · whisper <b>${esc(s.whisper)}</b> on <b>${esc(s.gpu || s.device)}</b>` +
      (s.keys.anthropic ? "" : ` · <span style="color:var(--yellow)">no ANTHROPIC key</span>`) +
      (s.keys.youtube ? "" : ` · <span style="color:var(--yellow)">no YOUTUBE key</span>`);
  } catch (e) { $("#sys").textContent = "server error"; }
}

/* ---------- screen 0: lookup ---------- */
function lookupScreen() {
  view.innerHTML = `<h1>Movie lookup</h1>
  <div class="card"><div class="row">
    <input id="lk-title" placeholder="Movie title…" style="flex:1;min-width:260px">
    <input id="lk-year" type="number" placeholder="year">
    <button id="lk-go">Look up</button>
  </div><p class="dim" style="margin-top:8px">How are other reactors faring with this title? Long-form reactions that are still up (and their age) are the best public signal of a tolerant rights holder.</p></div>
  <div id="lk-out"></div>`;
  const go = async () => {
    const title = $("#lk-title").value.trim(); if (!title) return;
    const year = $("#lk-year").value;
    $("#lk-out").innerHTML = `<div class="card dim">surveying YouTube…</div>`;
    try {
      const d = await api(`/lookup?title=${encodeURIComponent(title)}${year ? "&year="+year : ""}`);
      const sv = d.survey;
      $("#lk-out").innerHTML = `<div class="card">
        <div class="row"><h2 style="margin:0">${esc(title)}</h2><span class="flag ${sv.verdict}">${sv.verdict}</span></div>
        <p>${sv.n_longform} long-form reactions still up · ${sv.n_older_6mo} older than 6 months · median age ${sv.median_age_months ?? "–"} mo · oldest ${sv.oldest_age_months ?? "–"} mo · median views ${(sv.median_views ?? 0).toLocaleString()}</p>
        ${d.own.length ? `<p><b>This channel:</b> ${d.own.map(o => esc(o.outcome) + " (" + esc((o.at||"").slice(0,10)) + ")").join(", ")}</p>` : ""}
        <table><tr><th>age</th><th>length</th><th>views</th><th>channel</th><th>title</th></tr>
        ${sv.videos.map(v => `<tr><td>${v.age_months} mo</td><td>${v.minutes} min</td><td>${v.views.toLocaleString()}</td><td>${esc(v.channel)}</td><td class="dim">${esc(v.title)}</td></tr>`).join("")}
        </table>
        <div class="row" style="margin-top:12px"><button id="lk-new">Start a project with this title →</button></div>
      </div>`;
      $("#lk-new").onclick = () => { sessionStorage.setItem("newTitle", JSON.stringify({title, year})); location.hash = "#/new"; };
    } catch (e) { $("#lk-out").innerHTML = `<div class="card" style="color:var(--red)">${esc(e.message)}</div>`; }
  };
  $("#lk-go").onclick = go;
  $("#lk-title").onkeydown = e => e.key === "Enter" && go();
}

/* ---------- projects list + new ---------- */
async function projectsScreen() {
  const ps = await api("/projects");
  view.innerHTML = `<h1>Projects</h1>
  <div class="row" style="margin-bottom:14px"><button onclick="location.hash='#/new'">+ New project</button></div>
  <div class="grid">${ps.map(p => `
    <div class="card">
      <div class="row"><h2 style="margin:0">${esc(p.name)}</h2><span class="dim">${p.duration_min} min</span></div>
      <p class="dim mono" style="margin:6px 0">${esc((p.source||"").split(/[\\/]/).pop())}</p>
      <p class="dim">stages: ${Object.keys(p.stages).length ? Object.keys(p.stages).map(esc).join(", ") : "none yet"}</p>
      <div class="row" style="margin-top:8px">
        <button class="ghost" onclick="location.hash='#/p/${esc(p.name)}'">Open</button>
        ${p.renders.some(r => r.includes("preview")) ? `<button class="ghost" onclick="location.hash='#/review/${esc(p.name)}'">Review</button>` : ""}
      </div>
    </div>`).join("") || `<p class="dim">No projects yet — start from the Movie Lookup.</p>`}</div>`;
}

function newProjectScreen() {
  const pre = JSON.parse(sessionStorage.getItem("newTitle") || "{}");
  view.innerHTML = `<h1>New project</h1><div class="card">
    <form class="settings" id="np">
      <label>Project name<input name="name" required placeholder="stand-by-me"></label>
      <label>Recording file (path)<input name="source" required placeholder="samples/recording.mp4" style="min-width:340px"></label>
      <label>Movie title<input name="title" value="${esc(pre.title||"")}"></label>
      <label>Year<input name="year" type="number" value="${esc(pre.year||"")}"></label>
      <label>Studio<input name="studio" placeholder="e.g. Columbia Pictures"></label>
      <label>Reactor config<input name="reactor_config" value="configs/reactors/example.json"></label>
    </form>
    <div class="row" style="margin-top:14px"><button id="np-go">Create project</button></div>
  </div>`;
  $("#np-go").onclick = async () => {
    const f = Object.fromEntries(new FormData($("#np")));
    f.year = f.year ? +f.year : null;
    if (!f.name || !f.source) return toast("name and recording file are required", true);
    try {
      await api("/projects", { method: "POST", body: JSON.stringify(f) });
      sessionStorage.removeItem("newTitle");
      location.hash = "#/p/" + f.name;
    } catch (e) { toast(e.message, true); }
  };
}

/* ---------- project dashboard ---------- */
const STAGES = [
  ["detect",   "1 · Layout",    "find the movie & facecam regions"],
  ["analyze",  "2 · Analyze",   "transcribe, speakers, peaks, music, scenes"],
  ["narrative","3 · Narrative", "beat sheet + fan-favorite moments (Claude)"],
  ["card",     "3b · Title card","fetch movie logo, compose card"],
  ["select",   "4 · Select",    "build the cut (EDL)"],
  ["render",   "5 · Preview",   "fast 480p preview render"],
];
let pollTimer = null;

async function projectScreen(name) {
  const d = await api("/projects/" + name);
  const st = d.state.stages || {};
  const stageDone = k => ({detect:"layout",analyze:"speakers",narrative:"narrative",card:null,select:"select",render:"render_preview"}[k] in st || (k==="analyze" && ("transcribe" in st)));
  view.innerHTML = `<h1>${esc(name)} <span class="dim" style="font-size:13px">${(d.state.probe?.duration/60||0).toFixed(1)} min source</span></h1>
  <div class="row" style="align-items:flex-start">
    <div style="flex:2;min-width:420px">
      <div class="card" id="stages">
        ${STAGES.map(([k, label, hint]) => `
          <div class="stage ${stageDone(k) ? "done" : ""}" id="st-${k}">
            <span class="dot"></span><span class="name">${label}</span>
            <span class="status">${esc(hint)}</span>
            <div class="bar" style="display:none"><i style="width:0%"></i></div>
            <button class="small ghost" data-run="${k}">Run</button>
          </div>`).join("")}
        <div class="row" style="margin-top:10px">
          <button data-run="__chain">▶ Run all remaining</button>
          <button class="ghost" onclick="location.hash='#/review/${esc(name)}'">Open review</button>
          <button class="ghost" data-run="final">Final render (1080p)</button>
        </div>
      </div>
      <div class="card" id="joblog"><h2 style="margin-top:0">Activity</h2><div id="joblist" class="mono dim">idle</div></div>
    </div>
    <div style="flex:1;min-width:320px">
      ${d.layout_debug ? `<div class="card"><h2 style="margin-top:0">Detected layout</h2><img class="thumb" src="${d.layout_debug}?t=${Date.now()}"></div>` : ""}
      ${d.preflight ? `<div class="card"><h2 style="margin-top:0">Preflight</h2><span class="flag ${d.preflight.verdict}">${d.preflight.verdict}</span> <span class="dim">${d.preflight.n_longform} surviving long-form reactions</span></div>` : ""}
      ${d.narrative ? `<div class="card"><h2 style="margin-top:0">Narrative</h2><p class="dim">${d.narrative.n_beats} beats · ${d.narrative.n_key_lines} key lines · ${(d.narrative.moments||[]).length} moments</p>
        <details><summary>show</summary><table>${(d.narrative.beats||[]).map(b => `<tr><td>${fmtT(b.t0)}</td><td>${esc(b.priority)}</td><td>${esc(b.label)}</td></tr>`).join("")}</table></details></div>` : ""}
      <div class="card"><h2 style="margin-top:0">Renders</h2>
        ${d.renders.map(r => `<p><a href="${r.url}" target="_blank" style="color:var(--accent)">${esc(r.name)}</a> <span class="dim">${r.size_mb} MB</span><br>
          <span class="dim mono">${Object.entries(r.sidecars).map(([k,u]) => `<a href="${u}" target="_blank" style="color:var(--dim)">${esc(k)}</a>`).join(" · ")}</span></p>`).join("") || `<p class="dim">none yet</p>`}
        <div class="row"><button class="small ghost" id="oc-btn">Log claim outcome…</button><select id="oc-sel"><option>none</option><option>sharing</option><option>redirect</option><option>block</option></select></div>
      </div>
    </div>
  </div>`;
  $("#oc-btn").onclick = async () => {
    try { await api(`/projects/${name}/outcome`, { method: "POST", body: JSON.stringify({ outcome: $("#oc-sel").value }) }); toast("outcome recorded — the per-studio table just got smarter"); }
    catch (e) { toast(e.message, true); }
  };
  view.querySelectorAll("[data-run]").forEach(b => b.onclick = () => runStage(name, b.dataset.run));
  pollJobs(name);
}

async function runStage(name, stage) {
  const chain = stage === "__chain";
  const finalRender = stage === "final";
  const seq = chain ? STAGES.map(s => s[0]) : [finalRender ? "render" : stage];
  for (const st of seq) {
    try {
      const opts = st === "render" ? { preview: !finalRender } : {};
      const job = await api(`/projects/${name}/run/${st}`, { method: "POST", body: JSON.stringify({ options: opts }) });
      toast(`${st} started`);
      await waitJob(name, job.id, st);
    } catch (e) { toast(`${st}: ${e.message}`, true); break; }
  }
  if (location.hash === `#/p/${name}`) projectScreen(name);
}

function waitJob(name, id, stage) {
  return new Promise(resolve => {
    const el = $(`#st-${stage}`);
    const iv = setInterval(async () => {
      try {
        const j = await api("/jobs/" + id);
        if (el) {
          el.classList.toggle("running", j.state === "running");
          const bar = $(".bar", el); const status = $(".status", el);
          if (bar) { bar.style.display = "block"; $("i", bar).style.width = (j.frac * 100) + "%"; }
          if (status) status.textContent = j.msg || j.state;
        }
        if (j.state === "done" || j.state === "error") {
          clearInterval(iv);
          if (j.state === "error") toast(`${stage} failed: ${j.error}`, true); else toast(`${stage} done`);
          resolve(j);
        }
      } catch { clearInterval(iv); resolve(null); }
    }, 1200);
  });
}

async function pollJobs(name) {
  clearInterval(pollTimer);
  const tick = async () => {
    if (!location.hash.includes(name)) { clearInterval(pollTimer); return; }
    try {
      const js = await api("/jobs?project=" + name);
      const el = $("#joblist");
      if (el) el.innerHTML = js.slice(0, 6).map(j =>
        `<div>[${esc(j.state)}] ${esc(j.kind)} ${(j.frac*100|0)}% — ${esc(j.msg || "")}</div>`).join("") || "idle";
    } catch {}
  };
  pollTimer = setInterval(tick, 1500); tick();
}

/* ---------- review ---------- */
async function reviewScreen(name) {
  const d = await api("/projects/" + name);
  const prev = d.renders.find(r => r.name.includes("preview"));
  if (!prev) { view.innerHTML = `<div class="card">No preview render yet — run the pipeline first.</div>`; return; }
  let edl;
  try { edl = await api(`/projects/${name}/edl`); } catch { edl = null; }
  const pending = { drop: new Set(), flip: new Set() };
  view.innerHTML = `<h1>Review — ${esc(name)}</h1>
  <div class="row" style="align-items:flex-start">
    <div style="flex:3;min-width:480px">
      <video id="vid" src="${prev.url}" controls preload="metadata"></video>
      <div class="row" style="margin-top:10px">
        <button class="ghost small" id="mark-start">Film starts at playhead</button>
        <button class="ghost small" id="mark-end">Film ends at playhead</button>
        <span class="dim" id="bounds-msg">${d.state.film_bounds ? "bounds: " + d.state.film_bounds.map(x => fmtT(x)).join(" → ") + " (source)" : ""}</span>
      </div>
      <div class="card" style="margin-top:10px"><h2 style="margin-top:0">Pending changes</h2>
        <p class="dim" id="pend">none</p>
        <div class="row">
          <button id="save-edl" disabled>Save EDL</button>
          <button class="ghost" id="rerender">Re-render preview</button>
          <button class="ghost" id="reselect">Re-run select (discard manual edits)</button>
        </div>
      </div>
    </div>
    <div style="flex:2;min-width:380px;max-height:78vh;overflow:auto" class="card">
      <h2 style="margin-top:0">Cut list ${edl ? `<span class="dim">(${edl.segments.length} segments, ${(edl.duration/60).toFixed(1)} min)</span>` : ""}</h2>
      ${edl ? `<table id="segs"><tr><th>at</th><th>layout</th><th>note</th><th></th></tr>
        ${edl.segments.map(s => `<tr data-id="${esc(s.id)}">
          <td><a href="#" data-seek="${s.at}" style="color:var(--accent)">${fmtT(s.at)}</a></td>
          <td><span class="lay ${esc(s.layout)}">${esc(s.layout.replace("-large",""))}</span></td>
          <td class="dim" style="font-size:11px">${esc((s.note || s.kind).slice(0, 60))}</td>
          <td class="seg-actions">
            <button class="small ghost" data-act="flip" title="flip layout">⇄</button>
            <button class="small ghost" data-act="drop" title="drop segment">✕</button>
          </td></tr>`).join("")}</table>` : `<p class="dim">no EDL</p>`}
    </div>
  </div>`;
  const vid = $("#vid");
  view.querySelectorAll("[data-seek]").forEach(a => a.onclick = e => { e.preventDefault(); vid.currentTime = +a.dataset.seek; vid.play(); });
  const refreshPend = () => {
    $("#pend").textContent = (pending.drop.size || pending.flip.size)
      ? `drop ${pending.drop.size}, flip ${pending.flip.size}` : "none";
    $("#save-edl").disabled = !(pending.drop.size || pending.flip.size);
  };
  view.querySelectorAll("[data-act]").forEach(b => b.onclick = () => {
    const id = b.closest("tr").dataset.id;
    const set = pending[b.dataset.act];
    set.has(id) ? set.delete(id) : set.add(id);
    b.closest("tr").style.opacity = pending.drop.has(id) ? .35 : 1;
    refreshPend();
  });
  $("#save-edl").onclick = async () => {
    try {
      await api(`/projects/${name}/edl`, { method: "POST", body: JSON.stringify({ drop: [...pending.drop], flip: [...pending.flip] }) });
      toast("EDL saved — re-render the preview to see it"); pending.drop.clear(); pending.flip.clear(); refreshPend();
    } catch (e) { toast(e.message, true); }
  };
  let markStart = null;
  $("#mark-start").onclick = () => { markStart = vid.currentTime; $("#bounds-msg").textContent = `start marked at ${fmtT(markStart)} — now mark the end`; };
  $("#mark-end").onclick = async () => {
    if (markStart == null) return toast("mark the start first", true);
    try {
      const r = await api(`/projects/${name}/film-bounds`, { method: "POST", body: JSON.stringify({ start: markStart, end: vid.currentTime, from_preview: true }) });
      $("#bounds-msg").textContent = "bounds: " + r.film_bounds.map(x => fmtT(x)).join(" → ") + " (source) — re-run select";
      toast("film bounds pinned");
    } catch (e) { toast(e.message, true); }
  };
  $("#rerender").onclick = () => runStage(name, "render");
  $("#reselect").onclick = () => runStage(name, "select");
}

/* ---------- settings ---------- */
const HIDDEN = new Set(["layout_template"]);
function fieldFor(key, val, schema, prefix) {
  const id = prefix + "." + key;
  if (typeof val === "boolean")
    return `<label>${esc(key)}<select data-k="${id}"><option ${val ? "selected":""}>true</option><option ${!val ? "selected":""}>false</option></select></label>`;
  if (typeof val === "number")
    return `<label>${esc(key)}<input data-k="${id}" type="number" step="any" value="${val}"></label>`;
  if (val === null || typeof val === "string")
    return `<label>${esc(key)}<input data-k="${id}" value="${esc(val ?? "")}"></label>`;
  if (Array.isArray(val))
    return `<label>${esc(key)} <span class="dim">(comma list)</span><input data-k="${id}" data-list="1" value="${esc(val.join(", "))}"></label>`;
  return `<fieldset><legend>${esc(key)}</legend>${Object.entries(val).filter(([k]) => !HIDDEN.has(k)).map(([k, v]) => fieldFor(k, v, schema, id)).join("")}</fieldset>`;
}
async function settingsScreen() {
  const rPath = "configs/reactors/example.json";
  const tPath = "configs/titles/example.json";
  const [r, t] = await Promise.all([api(`/settings/reactor?path=${rPath}`), api(`/settings/title?path=${tPath}`)]);
  const render = (kind, data) => `<div class="card"><h2 style="margin-top:0">${kind === "reactor" ? "Channel (reactor)" : "Movie (title)"} settings <span class="dim mono">${esc(data.path)}</span></h2>
    <form class="settings" id="f-${kind}">${Object.entries(data.values).filter(([k]) => !HIDDEN.has(k)).map(([k, v]) => fieldFor(k, v, data.schema, "")).join("")}</form>
    <div class="row" style="margin-top:12px"><button data-save="${kind}" data-path="${esc(data.path)}">Save</button></div></div>`;
  view.innerHTML = `<h1>Settings</h1>` + render("reactor", r) + render("title", t);
  view.querySelectorAll("[data-save]").forEach(btn => btn.onclick = async () => {
    const kind = btn.dataset.save;
    const values = {};
    view.querySelectorAll(`#f-${kind} [data-k]`).forEach(inp => {
      const keys = inp.dataset.k.split(".").filter(Boolean);
      let o = values;
      keys.slice(0, -1).forEach(k => o = (o[k] ??= {}));
      let v = inp.value;
      if (inp.dataset.list) v = v.split(",").map(x => x.trim()).filter(Boolean).map(x => isNaN(+x) ? x : +x);
      else if (inp.type === "number") v = v === "" ? null : +v;
      else if (inp.tagName === "SELECT" && (v === "true" || v === "false")) v = v === "true";
      else if (v === "") v = null;
      o[keys.at(-1)] = v;
    });
    try { await api("/settings/" + kind, { method: "POST", body: JSON.stringify({ path: btn.dataset.path, values }) }); toast("saved"); }
    catch (e) { toast(e.message, true); }
  });
}

/* ---------- router ---------- */
async function route() {
  clearInterval(pollTimer);
  const h = location.hash || "#/lookup";
  document.querySelectorAll("nav a").forEach(a => a.classList.toggle("active", h.startsWith(a.hash || a.getAttribute("href"))));
  try {
    if (h.startsWith("#/lookup")) lookupScreen();
    else if (h.startsWith("#/projects")) await projectsScreen();
    else if (h.startsWith("#/new")) newProjectScreen();
    else if (h.startsWith("#/p/")) await projectScreen(decodeURIComponent(h.slice(4)));
    else if (h.startsWith("#/review/")) await reviewScreen(decodeURIComponent(h.slice(9)));
    else if (h.startsWith("#/settings")) await settingsScreen();
    else lookupScreen();
  } catch (e) { view.innerHTML = `<div class="card" style="color:var(--red)">${esc(e.message)}</div>`; }
}
window.addEventListener("hashchange", route);
route(); sysBadge();
