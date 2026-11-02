/* Tarbiyat frontend — plain JS, no build step, talks to the FastAPI app on the same origin. */

"use strict";

const state = {
  token: localStorage.getItem("tarbiyat_token") || "",
  user: null,
  datasets: [],
  active: null,
  activeProfile: [],
  lastRun: null,
};

/* ── Tiny DOM helpers ──────────────────────────────────────────────── */
const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

function log(msg, kind) {
  const line = el("div", kind || "", msg);
  $("#run-log").prepend(line);
  while ($("#run-log").children.length > 40) {
    $("#run-log").lastChild.remove();
  }
}

function showError(target, message) {
  const node = typeof target === "string" ? $(target) : target;
  if (!node) return;
  node.textContent = message || "";
  node.classList.toggle("hidden", !message);
}

/* ── API ───────────────────────────────────────────────────────────── */
async function api(path, options = {}) {
  const headers = Object.assign({}, options.headers || {});
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  if (options.body && !(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  const res = await fetch(path, Object.assign({}, options, { headers }));
  if (res.status === 401) {
    setSignedOut();
    throw new Error("Session expired. Please sign in again.");
  }
  if (res.status === 204) return null;

  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch (_) {
    data = null;
  }

  if (!res.ok) {
    let detail = (data && data.detail) || `Request failed (${res.status})`;
    if (Array.isArray(detail)) {
      detail = detail
        .map((d) => (d.loc ? `${d.loc.join(".")}: ${d.msg}` : d.msg || String(d)))
        .join("; ");
    }
    throw new Error(detail);
  }
  return data;
}

/* ── Auth ──────────────────────────────────────────────────────────── */
function setSignedIn(user) {
  state.user = user;
  $("#auth-view").classList.add("hidden");
  $("#app-view").classList.remove("hidden");
  $("#user-email").textContent = user.email;
  loadDatasets();
}

function setSignedOut() {
  state.token = "";
  state.user = null;
  localStorage.removeItem("tarbiyat_token");
  $("#app-view").classList.add("hidden");
  $("#auth-view").classList.remove("hidden");
}

function afterAuth(data) {
  state.token = data.access_token;
  localStorage.setItem("tarbiyat_token", state.token);
  setSignedIn(data.user);
  log(`signed in as ${data.user.email}`, "ok");
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    const which = tab.dataset.tab;
    $("#login-form").classList.toggle("hidden", which !== "login");
    $("#register-form").classList.toggle("hidden", which !== "register");
    showError("#auth-error", "");
  });
});

$("#login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  showError("#auth-error", "");
  const form = new FormData(event.target);
  try {
    const data = await api("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({
        email: form.get("email"),
        password: form.get("password"),
      }),
    });
    afterAuth(data);
  } catch (err) {
    showError("#auth-error", err.message);
  }
});

$("#register-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  showError("#auth-error", "");
  const form = new FormData(event.target);
  try {
    const data = await api("/api/auth/register", {
      method: "POST",
      body: JSON.stringify({
        email: form.get("email"),
        password: form.get("password"),
        display_name: form.get("display_name") || "",
      }),
    });
    afterAuth(data);
  } catch (err) {
    showError("#auth-error", err.message);
  }
});

$("#logout").addEventListener("click", () => {
  setSignedOut();
  log("signed out");
});

/* ── Datasets ──────────────────────────────────────────────────────── */
async function loadDatasets() {
  try {
    state.datasets = await api("/api/datasets");
    renderDatasets();
  } catch (err) {
    log(`datasets: ${err.message}`, "err");
  }
}

function renderDatasets() {
  const list = $("#dataset-list");
  list.innerHTML = "";
  if (!state.datasets.length) {
    const li = el("li", "ds-sub", "No datasets yet. Upload a CSV to start.");
    li.style.padding = "0.6rem";
    list.append(li);
    return;
  }
  state.datasets.forEach((ds) => {
    const li = el("li", "dataset-item");
    if (state.active && state.active.id === ds.id) li.classList.add("active");
    li.append(el("span", "ds-title", ds.name));
    li.append(
      el("span", "ds-sub", `${ds.row_count} rows · ${ds.column_count} cols · ${ds.task}`)
    );
    li.addEventListener("click", () => selectDataset(ds));
    list.append(li);
  });
}

$("#toggle-upload").addEventListener("click", () => {
  $("#upload-form").classList.toggle("hidden");
});

$("#upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(event.target);
  const body = new FormData();
  body.append("file", form.get("file"));
  body.append("target", form.get("target"));
  body.append("name", form.get("name") || "");
  try {
    const ds = await api("/api/datasets", { method: "POST", body });
    log(`uploaded ${ds.name} (${ds.row_count} rows, ${ds.task})`, "ok");
    $("#upload-form").reset();
    $("#upload-form").classList.add("hidden");
    await loadDatasets();
    selectDataset(ds);
  } catch (err) {
    log(`upload failed: ${err.message}`, "err");
  }
});

async function selectDataset(ds) {
  state.active = ds;
  state.activeProfile = ds.profile || [];
  renderDatasets();

  $("#empty-state").classList.add("hidden");
  $("#work").classList.remove("hidden");
  $("#ds-name").textContent = ds.name;
  $("#ds-meta").textContent =
    `${ds.row_count} rows · ${ds.column_count} columns · target "${ds.target}" · fp ${ds.fingerprint}`;
  const taskTag = $("#ds-task");
  taskTag.textContent = ds.task;
  taskTag.className = "tag" + (ds.task === "regression" ? " regression" : "");

  // Regression cannot use cross-entropy, so drop those options rather than let the user
  // pick something the server will refuse.
  const lossSelect = $("#cfg-loss");
  Array.from(lossSelect.options).forEach((opt) => {
    if (!opt.value) return;
    const isClassLoss = opt.value === "cce" || opt.value === "bce";
    opt.disabled = ds.task === "regression" ? isClassLoss : opt.value === "mse";
  });
  lossSelect.value = "";

  $("#metrics").classList.add("hidden");
  $("#confusion").classList.add("hidden");
  $("#predict-box").classList.add("hidden");
  drawChart([]);
  showError("#train-error", "");
  log(`selected ${ds.name}`);
}

/* ── Training ──────────────────────────────────────────────────────── */
function parseLayers(raw) {
  return raw
    .split(",")
    .map((s) => parseInt(s.trim(), 10))
    .filter((n) => !Number.isNaN(n) && n > 0);
}

$("#train-btn").addEventListener("click", async () => {
  if (!state.active) return;
  showError("#train-error", "");
  const btn = $("#train-btn");
  btn.disabled = true;
  btn.textContent = "Training…";

  const payload = {
    dataset_id: state.active.id,
    hidden_layers: parseLayers($("#cfg-layers").value),
    activation: $("#cfg-activation").value,
    init: $("#cfg-init").value,
    optimizer: $("#cfg-optimizer").value,
    loss: $("#cfg-loss").value || null,
    learning_rate: parseFloat($("#cfg-lr").value),
    epochs: parseInt($("#cfg-epochs").value, 10),
    batch_size: parseInt($("#cfg-batch").value, 10),
    l2: parseFloat($("#cfg-l2").value),
    test_fraction: parseFloat($("#cfg-split").value),
    early_stopping: $("#cfg-early").checked,
    seed: parseInt($("#cfg-seed").value, 10),
  };

  if (!payload.hidden_layers.length) {
    showError("#train-error", "Give at least one hidden layer width, e.g. 16 or 32,16.");
    btn.disabled = false;
    btn.textContent = "Train model";
    return;
  }

  try {
    const run = await api("/api/runs", { method: "POST", body: JSON.stringify(payload) });
    state.lastRun = run;
    renderRun(run);
    log(
      `run #${run.id} done in ${run.duration_ms} ms · ${run.param_count} params · loss ${run.loss}`,
      "ok"
    );
  } catch (err) {
    showError("#train-error", err.message);
    log(`train failed: ${err.message}`, "err");
  } finally {
    btn.disabled = false;
    btn.textContent = "Train model";
  }
});

function scoreClass(value, kind) {
  if (value === null || value === undefined) return "";
  if (kind === "r2") return value > 0.7 ? "good" : value > 0.35 ? "mid" : "bad";
  return value > 0.85 ? "good" : value > 0.6 ? "mid" : "bad";
}

function renderRun(run) {
  const m = run.metrics || {};
  const box = $("#metrics");
  box.innerHTML = "";
  box.classList.remove("hidden");

  const add = (label, value, cls) => {
    const card = el("div", "metric");
    card.append(el("div", "label", label));
    card.append(el("div", "value " + (cls || ""), value));
    box.append(card);
  };

  if (m.task === "classification") {
    add("test accuracy", (m.accuracy * 100).toFixed(1) + "%", scoreClass(m.accuracy));
    add("baseline", (m.majority_baseline * 100).toFixed(1) + "%", "");
    add("epochs run", String(run.history.length), "");
    add("params", run.param_count.toLocaleString(), "");
  } else {
    add("r²", m.r2.toFixed(3), scoreClass(m.r2, "r2"));
    add("rmse", m.rmse.toFixed(2), "");
    add("mae", m.mae.toFixed(2), "");
    add("baseline rmse", m.baseline_rmse.toFixed(2), "");
    // baseline_rmse is the constant predictor's error; beating it is the minimum bar.
    if (m.rmse > m.baseline_rmse) {
      log("warning: rmse is worse than predicting the mean — the model learned nothing", "err");
    }
  }

  drawChart(run.history);
  renderConfusion(m, run);
  buildPredictForm();
}

function renderConfusion(metrics, run) {
  const host = $("#confusion");
  host.innerHTML = "";
  if (metrics.task !== "classification") {
    host.classList.add("hidden");
    return;
  }
  host.classList.remove("hidden");

  const labels = metrics.labels || [];
  const matrix = metrics.confusion_matrix || [];
  const table = el("table");
  const head = el("tr");
  head.append(el("th", "", ""));
  labels.forEach((label) => head.append(el("th", "", label)));
  table.append(head);

  matrix.forEach((row, i) => {
    const tr = el("tr");
    tr.append(el("th", "", labels[i]));
    row.forEach((count, j) => {
      const td = el("td", i === j ? "diag" : "", String(count));
      tr.append(td);
    });
    table.append(tr);
  });

  host.append(table);

  if (metrics.per_class && metrics.per_class.length) {
    const caption = el("p", "hint");
    caption.textContent =
      "Per-class F1 — " +
      metrics.per_class.map((c) => `${c.label}: ${c.f1.toFixed(2)}`).join("  ·  ");
    host.append(caption);
  }
}

/* ── Loss chart (canvas, no library) ───────────────────────────────── */
function drawChart(history) {
  const canvas = $("#loss-chart");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = canvas.clientWidth || 720;
  const cssHeight = 260;
  canvas.width = cssWidth * dpr;
  canvas.height = cssHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  const pad = { top: 16, right: 14, bottom: 26, left: 46 };
  const w = cssWidth - pad.left - pad.right;
  const h = cssHeight - pad.top - pad.bottom;

  ctx.strokeStyle = "#22303f";
  ctx.fillStyle = "#6b7d91";
  ctx.font = "11px ui-monospace, monospace";
  ctx.lineWidth = 1;

  if (!history || history.length === 0) {
    ctx.fillText("Run a training job to see the loss curve.", pad.left, pad.top + h / 2);
    return;
  }

  const values = [];
  history.forEach((p) => {
    values.push(p.train_loss);
    if (p.val_loss !== null && p.val_loss !== undefined) values.push(p.val_loss);
  });
  const max = Math.max(...values);
  const min = Math.min(...values);
  const span = max - min || 1;

  const x = (i) => pad.left + (history.length === 1 ? w / 2 : (i / (history.length - 1)) * w);
  const y = (v) => pad.top + h - ((v - min) / span) * h;

  // Grid + axis labels
  for (let g = 0; g <= 4; g += 1) {
    const gy = pad.top + (h / 4) * g;
    ctx.beginPath();
    ctx.moveTo(pad.left, gy);
    ctx.lineTo(pad.left + w, gy);
    ctx.stroke();
    const value = max - (span / 4) * g;
    ctx.fillText(value.toFixed(3), 4, gy + 4);
  }
  ctx.fillText("epoch 1", pad.left, cssHeight - 8);
  ctx.fillText(String(history.length), pad.left + w - 22, cssHeight - 8);

  const line = (key, colour) => {
    ctx.strokeStyle = colour;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    let started = false;
    history.forEach((p, i) => {
      const v = p[key];
      if (v === null || v === undefined) return;
      if (!started) {
        ctx.moveTo(x(i), y(v));
        started = true;
      } else {
        ctx.lineTo(x(i), y(v));
      }
    });
    ctx.stroke();
  };

  line("train_loss", "#38bdf8");
  line("val_loss", "#f472b6");
}

window.addEventListener("resize", () => {
  if (state.lastRun) drawChart(state.lastRun.history);
});

/* ── Predict ───────────────────────────────────────────────────────── */
function buildPredictForm() {
  const host = $("#predict-fields");
  host.innerHTML = "";
  const features = state.activeProfile.filter((c) => c.kind !== "target");
  if (!features.length) return;

  features.forEach((col) => {
    const label = el("label", "", col.name);
    let input;
    if (col.kind === "categorical" && col.categories && col.categories.length) {
      input = document.createElement("select");
      col.categories.forEach((cat) => {
        const opt = document.createElement("option");
        opt.value = cat;
        opt.textContent = cat;
        input.append(opt);
      });
    } else {
      input = document.createElement("input");
      input.type = "number";
      input.step = "any";
      input.placeholder = col.mean !== null && col.mean !== undefined ? col.mean.toFixed(2) : "0";
    }
    input.dataset.column = col.name;
    label.append(input);
    host.append(label);
  });

  $("#predict-box").classList.remove("hidden");
  $("#predict-out").textContent = "";
}

$("#predict-btn").addEventListener("click", async () => {
  if (!state.lastRun) return;
  const row = {};
  $("#predict-fields")
    .querySelectorAll("[data-column]")
    .forEach((input) => {
      row[input.dataset.column] = input.type === "number" ? parseFloat(input.value || "0") : input.value;
    });

  try {
    const res = await api(`/api/runs/${state.lastRun.id}/predict`, {
      method: "POST",
      body: JSON.stringify({ rows: [row] }),
    });
    const p = res.predictions[0];
    let text = `→ ${p.predicted}`;
    if (p.confidence !== null && p.confidence !== undefined) {
      text += ` (${(p.confidence * 100).toFixed(1)}% confident)`;
    }
    const probs = Object.entries(p.probabilities || {});
    if (probs.length > 1) {
      text += "\n" + probs.map(([k, v]) => `  ${k}: ${(v * 100).toFixed(1)}%`).join("\n");
    }
    $("#predict-out").textContent = text;
    $("#predict-out").style.whiteSpace = "pre";
    log(`predicted ${p.predicted}`, "ok");
  } catch (err) {
    $("#predict-out").textContent = err.message;
    log(`predict failed: ${err.message}`, "err");
  }
});

/* ── Boot ──────────────────────────────────────────────────────────── */
(async function boot() {
  try {
    const health = await api("/api/health");
    $("#api-tag").textContent = `v${health.version}`;
  } catch (_) {
    $("#api-tag").textContent = "offline";
  }

  if (!state.token) return;
  try {
    const user = await api("/api/auth/me");
    setSignedIn(user);
  } catch (_) {
    setSignedOut();
  }
})();
