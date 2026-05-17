// Research Agent SPA. Vanilla JS, chat-style layout.
// Left column = conversational chat log. Right column = TODO + activity panels.

const $ = (id) => document.getElementById(id);

const els = {
  goal: $("goal-input"),
  run: $("run-button"),
  fileBtn: $("dz-button"),
  fileInput: $("file-input"),
  chips: $("file-chips"),
  status: $("status-strip"),
  badge: $("session-badge"),
  chatLog: $("chat-log"),
  emptyChat: $("empty-chat"),
  taskList: $("task-list"),
  count: $("todo-count"),
  completeBadge: $("complete-badge"),
  feed: $("activity-feed"),
  pause: $("pause-scroll"),
};

const state = {
  sessionId: null,
  evtSource: null,
  tasks: new Map(),
  pendingFiles: [],
  autoScroll: true,
  statusBubble: null,   // the current "thinking" bubble (gets replaced by report)
};

// --- composer behavior -------------------------------------------------

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

// Send on click; also on Ctrl/Cmd + Enter.
els.run.addEventListener("click", send);
els.goal.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    send();
  }
});

// Auto-resize textarea
els.goal.addEventListener("input", () => {
  els.goal.style.height = "auto";
  els.goal.style.height = Math.min(els.goal.scrollHeight, 160) + "px";
});

// --- send action -------------------------------------------------------

async function send() {
  const goal = els.goal.value.trim();
  if (!goal) {
    setStatus("Please enter a goal.", "warn");
    return;
  }

  resetForNewSession();
  pushUserMessage(goal, state.pendingFiles.map((f) => f.name));
  state.statusBubble = pushStatusBubble("Creating session");

  setRunning(true);
  setStatus("Creating session…");

  const fd = new FormData();
  fd.append("goal", goal);
  for (const f of state.pendingFiles) fd.append("files", f, f.name);

  let resp;
  try {
    resp = await fetch("/api/sessions", { method: "POST", body: fd });
  } catch (e) {
    failStatusBubble(`Network error: ${e.message}`);
    setStatus(`Network error: ${e.message}`, "error");
    setRunning(false);
    return;
  }

  if (!resp.ok) {
    const detail = await resp.text();
    failStatusBubble(`Failed: ${detail}`);
    setStatus(`Failed to create session.`, "error");
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

  // Clear pending files after successful upload.
  state.pendingFiles = [];
  renderChips();
  els.goal.value = "";
  els.goal.style.height = "auto";

  updateStatusBubble("Planning");
  setStatus("Planning…");
  subscribe(body.session_id);
}

// --- SSE subscription --------------------------------------------------

function subscribe(sessionId) {
  if (state.evtSource) state.evtSource.close();
  const es = new EventSource(`/api/sessions/${sessionId}/events`);
  state.evtSource = es;

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

// --- event dispatch ----------------------------------------------------

function dispatch(type, event) {
  const data = event.data || {};
  switch (type) {
    case "session.started":
      appendFeed("status", "session", "started");
      break;
    case "plan.ready":
      renderPlan(data.tasks || []);
      updateStatusBubble(`Planning complete — ${data.tasks?.length || 0} tasks`);
      setStatus(`Plan ready — ${data.tasks?.length || 0} tasks.`);
      break;
    case "task.status_changed":
      updateTaskStatus(data.task_id, data.status, data.error);
      setStatus(progressText());
      updateStatusBubble(progressText());
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
      appendFeed("warn", data.tool, data.error || "retrying");
      appendTaskToolLine(data.task_id, data.tool, "retrying", "warn");
      break;
    case "report.ready":
      removeStatusBubble();
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
      removeStatusBubble();
      pushAgentError(data.error || "session failed");
      setStatus(`Failed: ${data.error || "unknown error"}`, "error");
      setRunning(false);
      break;
    case "guardrail.triggered":
      appendFeed("error", "guardrail", `${data.check}: ${data.reason || ""}`);
      break;
  }
}

// --- chat messages -----------------------------------------------------

function pushUserMessage(text, filenames) {
  hideEmptyChat();
  const div = document.createElement("div");
  div.className = "msg user";
  let html = escapeHtml(text);
  if (filenames && filenames.length) {
    html += `<div style="margin-top:6px;font-size:12px;opacity:0.85;">📎 ${filenames.map(escapeHtml).join(", ")}</div>`;
  }
  div.innerHTML = html;
  els.chatLog.appendChild(div);
  scrollChatToBottom();
}

function pushStatusBubble(text) {
  hideEmptyChat();
  const div = document.createElement("div");
  div.className = "msg agent status";
  div.innerHTML = `<span class="status-text">${escapeHtml(text)}</span>` +
                  `<span class="typing"><span></span><span></span><span></span></span>`;
  els.chatLog.appendChild(div);
  scrollChatToBottom();
  return div;
}

function updateStatusBubble(text) {
  if (!state.statusBubble) {
    state.statusBubble = pushStatusBubble(text);
    return;
  }
  const span = state.statusBubble.querySelector(".status-text");
  if (span) span.textContent = text;
}

function failStatusBubble(text) {
  if (state.statusBubble) {
    state.statusBubble.classList.remove("status");
    state.statusBubble.classList.add("error");
    state.statusBubble.innerHTML = escapeHtml(text);
    state.statusBubble = null;
  } else {
    pushAgentError(text);
  }
}

function removeStatusBubble() {
  if (state.statusBubble && state.statusBubble.parentElement) {
    state.statusBubble.parentElement.removeChild(state.statusBubble);
  }
  state.statusBubble = null;
}

function pushAgentError(text) {
  hideEmptyChat();
  const div = document.createElement("div");
  div.className = "msg agent error";
  div.textContent = text;
  els.chatLog.appendChild(div);
  scrollChatToBottom();
}

function hideEmptyChat() {
  if (els.emptyChat) els.emptyChat.style.display = "none";
}

function scrollChatToBottom() {
  // Defer so the new node is laid out before we scroll.
  requestAnimationFrame(() => {
    els.chatLog.scrollTop = els.chatLog.scrollHeight;
  });
}

// --- report + verification ---------------------------------------------

function renderReport(markdown) {
  const div = document.createElement("div");
  div.className = "msg agent report";
  div.innerHTML = marked.parse(markdown);
  els.chatLog.appendChild(div);
  scrollChatToBottom();
}

function renderVerification(claims, notes) {
  if (!claims.length && !notes) return;
  const block = document.createElement("details");
  block.className = "verification-bubble";
  let inner = `<summary>⚠ Verification notes</summary>`;
  if (notes) inner += `<p>${escapeHtml(notes)}</p>`;
  if (claims.length) {
    inner += "<ul>";
    for (const c of claims) inner += `<li>${escapeHtml(c)}</li>`;
    inner += "</ul>";
  }
  block.innerHTML = inner;
  els.chatLog.appendChild(block);
  scrollChatToBottom();
}

// --- TODO panel --------------------------------------------------------

function renderPlan(tasks) {
  state.tasks.clear();
  els.taskList.innerHTML = "";
  els.count.textContent = tasks.length;
  els.completeBadge.hidden = true;
  for (const t of tasks) {
    state.tasks.set(t.id, { ...t });
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

  document.querySelectorAll(".task-row.active").forEach((r) => r.classList.remove("active"));
  if (status === "in_progress") row.classList.add("active");

  if (status === "failed" && error) {
    const lines = row.querySelector(".tool-lines");
    appendTagged(lines, "failed", "error", error);
  }
}

function appendTaskToolLine(taskId, tool, kind, cls) {
  const row = els.taskList.querySelector(`[data-task-id="${taskId}"]`);
  if (!row) return;
  const lines = row.querySelector(".tool-lines");
  appendTagged(lines, cls || "", tool, kind);
}

function appendTagged(container, cls, tool, label) {
  const div = document.createElement("div");
  div.className = `tool-line ${cls}`;
  div.innerHTML = `<span class="tool-name">${escapeHtml(tool)}</span> · ${escapeHtml(label)}`;
  container.appendChild(div);
}

// --- activity feed -----------------------------------------------------

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

// --- helpers -----------------------------------------------------------

function setStatus(text, kind) {
  els.status.textContent = text;
  els.status.style.color = {
    error: "var(--error)", warn: "var(--warn)", ok: "var(--success)",
  }[kind] || "var(--fg-muted)";
}

function setRunning(running) {
  els.run.disabled = running;
  els.fileBtn.disabled = running;
}

function progressText() {
  const total = state.tasks.size;
  if (!total) return "Planning…";
  let done = 0, current = null;
  for (const t of state.tasks.values()) {
    if (t.status === "done") done += 1;
    if (t.status === "in_progress") current = t;
  }
  if (current) {
    return `Task ${current.order_index + 1}/${total}: ${current.description.slice(0, 60)}…`;
  }
  if (done === total) return "Synthesizing report…";
  return `${done}/${total} tasks complete`;
}

function resetForNewSession() {
  state.tasks.clear();
  state.statusBubble = null;
  els.chatLog.innerHTML = "";
  els.emptyChat = null;  // gone for the rest of this session
  els.taskList.innerHTML = '<li class="empty-state">Planning…</li>';
  els.count.textContent = "0";
  els.completeBadge.hidden = true;
  els.feed.innerHTML = "";
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
