"use strict";
// All text from the server (OCR output) is inserted with textContent only, never innerHTML.
const $ = (id) => document.getElementById(id);
const state = { rows: [], modelLoaded: false };

function el(tag, props = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const kid of kids) node.append(kid);
  return node;
}

const NUM = /^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$/;
function parseValue(text) {
  const t = text.trim().replace(",", ".");
  if (t === "") return null;
  return NUM.test(t) ? Number(t) : text.trim();
}

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let msg = "Something went wrong.";
    try { msg = (await res.json()).detail || msg; } catch (_) {}
    throw new Error(typeof msg === "string" ? msg : "The request was not valid.");
  }
  return res;
}

function setStatus(text, bad = false) {
  const s = $("status");
  s.textContent = text;
  s.className = bad ? "bad" : "";
}

function refText(r) {
  const rr = r.reference_range;
  return rr ? (rr.raw || "") : "";
}

function renderRows() {
  const tbody = $("rows");
  tbody.replaceChildren();
  state.rows.forEach((row, i) => {
    const name = el("input", { type: "text", list: "known-tests", "aria-label": "Test name" });
    name.value = row.test_name || "";
    name.addEventListener("input", () => { row.test_name = name.value.trim() || null; });

    const val = el("input", { type: "text", inputmode: "decimal", "aria-label": "Value" });
    val.value = row.value_operator && row.value !== null ? `${row.value_operator}${row.value}` : (row.value ?? "");
    val.addEventListener("input", () => { row.value = parseValue(val.value); row.value_operator = null; });

    const unit = el("input", { type: "text", "aria-label": "Unit" });
    unit.value = row.unit || "";
    unit.addEventListener("input", () => { row.unit = unit.value.trim() || null; });

    const badge = el("span", { class: `badge ${row.validation_status}` });
    badge.textContent = row.validation_status.replace("_", " ");
    if (row.validation_warnings.length) badge.title = row.validation_warnings.join("\n");

    const del = el("button", { type: "button", "aria-label": "Remove row" });
    del.textContent = "✕";
    del.addEventListener("click", () => { state.rows.splice(i, 1); renderRows(); });

    const ref = el("td", { class: "ref" });
    ref.textContent = refText(row);
    tbody.append(el("tr", {}, el("td", {}, name), el("td", {}, val), el("td", {}, unit), ref, el("td", {}, badge), el("td", {}, del)));
  });
}

function payload() {
  return { results: state.rows, reviewed_all: $("confirm").checked };
}

function postJson(path, body) {
  return api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

async function download(fmt) {
  try {
    const res = await postJson(`/api/export/${fmt}`, payload());
    const url = URL.createObjectURL(await res.blob());
    const a = el("a", { href: url, download: `lab_results.${fmt}` });
    document.body.append(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  } catch (e) { setStatus(e.message, true); }
}

function renderPrediction(p) {
  const body = $("pred-body");
  body.replaceChildren();
  if (p.status !== "ok") {
    const h = el("p", { class: "big bad" }); h.textContent = "Not enough data to predict";
    const miss = el("p", {}); miss.textContent = "Missing or unusable values: " + p.missing_features.join(", ");
    body.append(h, miss);
  } else {
    const h = el("p", { class: "big" }); h.textContent = p.prediction;
    body.append(h);
    for (const [label, prob] of Object.entries(p.probabilities).sort((a, b) => b[1] - a[1])) {
      const bar = el("div", { class: "bar" }, el("span"));
      bar.firstChild.style.width = `${Math.round(prob * 100)}%`;
      const l = el("span"); l.textContent = label;
      const pct = el("span"); pct.textContent = `${(prob * 100).toFixed(1)}%`;
      body.append(el("div", { class: "prob" }, l, bar, pct));
    }
  }
  if (p.warnings.length) {
    const ul = el("ul", { class: "warnings" });
    p.warnings.forEach((w) => { const li = el("li"); li.textContent = w; ul.append(li); });
    body.append(ul);
  }
  const d = el("p", { class: "hint" }); d.textContent = p.disclaimer;
  body.append(d);
  $("prediction").hidden = false;
}

async function init() {
  try {
    const h = await (await api("/api/health")).json();
    state.modelLoaded = h.model_loaded;
    $("demo-banner").hidden = !h.demo;
    const mb = $("model-banner");
    if (!h.model_loaded) {
      mb.hidden = false;
      mb.textContent = h.model_error || "No prediction model is configured. You can still extract and download your values.";
    }
    const dl = $("known-tests");
    h.known_tests.forEach((t) => dl.append(el("option", { value: t })));
  } catch (e) { setStatus(e.message, true); }
}

const fileInput = $("file");
fileInput.addEventListener("change", () => {
  $("file-label").textContent = fileInput.files[0] ? fileInput.files[0].name : "Choose a PDF or image, or drop it here";
  $("process").disabled = !fileInput.files[0];
});
const drop = $("drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault(); drop.classList.remove("over");
  if (e.dataTransfer.files.length) { fileInput.files = e.dataTransfer.files; fileInput.dispatchEvent(new Event("change")); }
});

$("process").addEventListener("click", async () => {
  const f = fileInput.files[0];
  if (!f) return;
  const fd = new FormData(); fd.append("file", f);
  $("process").disabled = true; setStatus("Reading your report... this can take a moment.");
  try {
    const report = await (await api("/api/extract", { method: "POST", body: fd })).json();
    state.rows = report.results;
    renderRows();
    const ul = $("warnings"); ul.replaceChildren();
    report.warnings.forEach((w) => { const li = el("li"); li.textContent = w; ul.append(li); });
    $("review").hidden = false; $("prediction").hidden = true;
    $("confirm").checked = false; updatePredict();
    setStatus(report.results.length ? "" : "No values were detected. Try a clearer scan, or add rows by hand.");
  } catch (e) { setStatus(e.message, true); }
  finally { $("process").disabled = false; }
});

$("add-row").addEventListener("click", () => {
  state.rows.push({ test_name: null, original_test_name: null, raw_result_text: "", value: null, value_operator: null,
    unit: null, reference_range: null, abnormal_flag: null, specimen_type: null, page_number: 1, confidence: null,
    validation_status: "needs_review", validation_warnings: [], manually_reviewed: false });
  renderRows();
});

function updatePredict() { $("predict").disabled = !(state.modelLoaded && $("confirm").checked); }
$("confirm").addEventListener("change", updatePredict);

$("predict").addEventListener("click", async () => {
  setStatus("Running the model...");
  try {
    renderPrediction(await (await postJson("/api/predict", payload())).json());
    setStatus("");
  } catch (e) { setStatus(e.message, true); }
});
$("dl-json").addEventListener("click", () => download("json"));
$("dl-csv").addEventListener("click", () => download("csv"));
$("discard").addEventListener("click", () => {
  state.rows = []; fileInput.value = ""; $("file-label").textContent = "Choose a PDF or image, or drop it here";
  $("process").disabled = true; $("review").hidden = true; $("prediction").hidden = true; setStatus("");
});

init();
