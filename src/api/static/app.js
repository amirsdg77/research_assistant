// Research Agent SPA. Vanilla JS, no build step.
// Subscribes to SSE, dispatches by event.type, renders the TODO panel +
// activity feed + final report.

const $ = (id) => document.getElementById(id);

const els = {
  goal: $("goal-input"),
  run: $("run-button"),
  fileBtn: $("dz-button"),
  fileInput: $("file-input"),
  chips: $("file-chips"),
  status: $("status-strip"),
  badge: $("session-badge"),
  taskList: $("task-list"),
  count: $("todo-count"),
  completeBadge: $("complete-badge"),
  feed: $("activity-feed"),
  pause: $("pause-scroll"),
  reportSection: $("report-section"),
  reportBody: $("report-body"),
  verificationBlock: $("verification-block"),
  verificationBody: $("verification-body"),
};

// --- session state -------------------------------------------------------

const state = {
  sessionId: null,
  evtSource: null,
  tasks: new Map(),       // task_id -> { id, order_index, description, status, ... }
  pendingFiles: [],
  autoScroll: true,
};

// --- file attachment -----------------------------------------------------

els.fileBtn.addEventListener("click", () => els.fileInput.click());
els.fileInput.addEventListener("change", (e) => {
  for (const f of e.target.files) state.pendingFiles.push(f);
  renderChips();
  els.fileInput.value = "";
});

function renderChips() {
  els.chips.innerHTML = "";
  state.pendingFiles.forEach((f, i) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = f.name;
    const x = document.createElement("button");
    x.type = "button";
    x.textContent = "×";
    x.onclick = () => {
      state.pendingFiles.splice(i, 1);
      renderChips();
    };
    chip.appendChild(x);
    els.chips.appendChild(chip);
  });
}

// --- run button ----------------------------------------------------------

els.run.addEventListener("click", async () => {
  const goal = els.goal.value.trim();
  if (!goal) {
    setStatus("Please enter a goal.", "warn");
    return;
  }
  resetUiForNewSession();
  setRunning(true);
  setStatus("Creating session…");

  const fd = new FormData();
  fd.append("goal", goal);
  for (const f of state.pendingFiles) fd.append("files", f, f.name);

  let resp;
  try {
    resp = await fetch("/api/sessions", { method: "POST", body: fd });
  } catch (e) {
    setStatus(`Network error: ${e.message}`, "error");
    setRunning(false);
    return;
  }

  if (!resp.ok) {
    const detail = await resp.text();
    setStatus(`Failed to create session: ${detail}`, "error");
    setRunning(false);
    return;
  }
  const body = await resp.json();
  state.sessionId = body.session_id;
  els.badge.hidden = false;
  els.badge.textContent = `session ${body.session_id.slice(0, 8)}`;
  if (body.documents?.length) {
    appendFeed("status", "ingested", `${body.documents.length} doc(s) ingested`);
  }
  setStatus("Subscribing to events…");
  subscribe(body.session_id);
});

// --- SSE subscription ----------------------------------------------------

function subscribe(sessionId) {
  if (state.evtSource) state.evtSource.close();
  const es = new EventSource(`/api/sessions/${sessionId}/events`);
  state.evtSource = es;

  // We listen on specific event names rather than the default 'message' so
  // the server's `event: <type>` line routes here directly.
  const types = [
    "session.started", "plan.ready", "task.status_changed",
    "tool.invoked", "tool.completed", "tool.failed",
    "report.ready", "verification.ready",
    "session.completed", "session.failed",
    "guardrail.triggered",
  ];
  for (const t of types) {
    es.addEventListener(t, (e) => {
      let payload = {};
      try { payload = JSON.parse(e.data); } catch { /* ignore */ }
      dispatch(t, payload);
    });
  }
  es.onerror = () => {
    appendFeed("error", "sse", "event stream closed");
    es.close();
    setRunning(false);
  };
}

// --- event dispatch ------------------------------------------------------

function dispatch(type, event) {
  const data = event.data || {};
  switch (type) {
    case "session.started":
      setStatus("Planning…");
      appendFeed("status", "session", "started");
      break;
    case "plan.ready":
      renderPlan(data.tasks || []);
      setStatus(`Plan ready — ${data.tasks?.length || 0} tasks.`);
      break;
    case "task.status_changed":
      updateTaskStatus(data.task_id, data.status, data.error);
      setStatus(progressText());
      break;
    case "tool.invoked":
      appendFeed("tool", data.tool, `→ ${truncate(data.input_preview, 80)}`);
      appendTaskToolLine(data.task_id, data.tool, "invoked");
      break;
    case "tool.completed":
      appendFeed("tool", data.tool, "completed");
      appendTaskToolLine(data.task_id, data.tool, "completed");
      break;
    case "tool.failed":
      appendFeed("error", data.tool, data.error || "failed");
      appendTaskToolLine(data.task_id, data.tool, "failed");
      break;
    case "report.ready":
      renderReport(data.report || "");
      setStatus("Report ready. Verifying…");
      break;
    case "verification.ready":
      renderVerification(data.unsupported_claims || [], data.notes || "");
      break;
    case "session.completed":
      setStatus("Done.", "ok");
      els.completeBadge.hidden = false;
      setRunning(false);
      break;
    case "session.failed":
      setStatus(`Failed: ${data.error || "unknown error"}`, "error");
      setRunning(false);
      break;
    case "guardrail.triggered":
      appendFeed("error", "guardrail", `${data.check}: ${data.reason || ""}`);
      break;
  }
}

// --- TODO panel rendering ------------------------------------------------

function renderPlan(tasks) {
  state.tasks.clear();
  els.taskList.innerHTML = "";
  els.count.textContent = tasks.length;
  els.completeBadge.hidden = true;
  for (const t of tasks) {
    state.tasks.set(t.id, { ...t, tool_lines: [], sources: [], result_summary: null });
    els.taskList.appendChild(makeTaskRow(t));
  }
}

function makeTaskRow(t) {
  const li = document.createElement("li");
  li.className = `task-row ${t.status}`;
  li.dataset.taskId = t.id;
  li.innerHTML = `
    <div class="task-row-head">
      <span class="task-icon ${t.status}"></span>
      <span class="task-desc">${escapeHtml(t.description)}</span>
    </div>
    <div class="task-drawer">
      <div class="drawer-section">Tool activity</div>
      <div class="tool-lines"></div>
      <div class="drawer-section summary-section" hidden>Result</div>
      <div class="result-summary"></div>
      <div class="drawer-section sources-section" hidden>Sources</div>
      <div class="sources"></div>
    </div>`;
  li.querySelector(".task-row-head").addEventListener("click", () =>
    li.classList.toggle("expanded")
  );
  return li;
}

function updateTaskStatus(taskId, status, error) {
  const t = state.tasks.get(taskId);
  if (!t) return;
  t.status = status;
  const row = els.taskList.querySelector(`[data-task-id="${taskId}"]`);
  if (!row) return;

  row.className = `task-row ${status}`;
  const icon = row.querySelector(".task-icon");
  icon.className = `task-icon ${status}`;

  // Auto-highlight the in_progress task; clear when it advances.
  document.querySelectorAll(".task-row.active").forEach((r) => r.classList.remove("active"));
  if (status === "in_progress") row.classList.add("active");

  if (status === "failed" && error) {
    const lines = row.querySelector(".tool-lines");
    appendTagged(lines, "failed", "error", error);
  }
}

function appendTaskToolLine(taskId, tool, kind) {
  const row = els.taskList.querySelector(`[data-task-id="${taskId}"]`);
  if (!row) return;
  const lines = row.querySelector(".tool-lines");
  const cls = kind === "failed" ? "failed" : "";
  appendTagged(lines, cls, tool, kind);
}

function appendTagged(container, cls, tool, label) {
  const div = document.createElement("div");
  div.className = `tool-line ${cls}`;
  div.innerHTML = `<span class="tool-name">${escapeHtml(tool)}</span> · ${escapeHtml(label)}`;
  container.appendChild(div);
}

// --- activity feed -------------------------------------------------------

els.pause.addEventListener("change", () => {
  state.autoScroll = !els.pause.checked;
});

function appendFeed(kind, tag, msg) {
  const div = document.createElement("div");
  div.className = `feed-line ${kind}`;
  const ts = new Date().toLocaleTimeString("en-US", { hour12: false });
  div.innerHTML = `<span class="ts">${ts}</span><span class="tag">${escapeHtml(tag)}</span> ${escapeHtml(msg || "")}`;
  els.feed.appendChild(div);
  if (state.autoScroll) els.feed.scrollTop = els.feed.scrollHeight;
}

// --- report rendering ----------------------------------------------------

function renderReport(markdown) {
  els.reportSection.hidden = false;
  // marked v12+ doesn't auto-sanitize; we trust the synthesizer's output
  // since it's our own LLM. The model is constrained to a citation format
  // (numbered [n] references) so risk of arbitrary HTML is low.
  els.reportBody.innerHTML = marked.parse(markdown);
}

function renderVerification(claims, notes) {
  if (!claims.length && !notes) {
    els.verificationBlock.hidden = true;
    return;
  }
  els.verificationBlock.hidden = false;
  let html = notes ? `<p>${escapeHtml(notes)}</p>` : "";
  if (claims.length) {
    html += "<ul>";
    for (const c of claims) html += `<li>${escapeHtml(c)}</li>`;
    html += "</ul>";
  }
  els.verificationBody.innerHTML = html;
}

// --- helpers -------------------------------------------------------------

function setStatus(text, kind) {
  els.status.textContent = text;
  els.status.style.color = {
    error: "var(--error)", warn: "var(--warn)", ok: "var(--success)",
  }[kind] || "var(--fg-muted)";
}

function setRunning(running) {
  els.run.disabled = running;
  els.goal.disabled = running;
  els.fileBtn.disabled = running;
}

function progressText() {
  const total = state.tasks.size;
  if (!total) return "";
  let done = 0, current = null;
  for (const t of state.tasks.values()) {
    if (t.status === "done") done += 1;
    if (t.status === "in_progress") current = t;
  }
  if (current) {
    return `Running task ${current.order_index + 1} of ${total}: ${current.description.slice(0, 70)}…`;
  }
  return `${done} of ${total} tasks complete.`;
}

function resetUiForNewSession() {
  state.tasks.clear();
  els.taskList.innerHTML = '<li class="empty-state">Planning…</li>';
  els.count.textContent = "0";
  els.completeBadge.hidden = true;
  els.feed.innerHTML = "";
  els.reportSection.hidden = true;
  els.reportBody.innerHTML = "";
  els.verificationBlock.hidden = true;
  els.verificationBody.innerHTML = "";
  els.badge.hidden = true;
  if (state.evtSource) state.evtSource.close();
  state.evtSource = null;
}

function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function truncate(s, n) {
  s = String(s ?? "");
  return s.length > n ? s.slice(0, n) + "…" : s;
}
