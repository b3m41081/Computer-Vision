const views = ["comparison", "left", "disparity", "ground_truth", "error", "depth"];
let currentView = "comparison";
let lastPayload = null;
let tests = [];
let benchmarkSortMetric = "mae";

function $(id) {
  return document.getElementById(id);
}

function params() {
  return {
    algorithm: $("algorithm").value,
    block_size: Number($("blockSize").value),
    uniqueness_ratio: Number($("uniqueness").value),
    speckle_window_size: Number($("speckleWindow").value),
    speckle_range: Number($("speckleRange").value),
    min_disparity: Number($("minDisparity").value),
    num_disparities: Number($("numDisparities").value),
  };
}

function setBusy(isBusy) {
  ["compute", "save", "benchmark", "loadTest"].forEach((id) => {
    $(id).disabled = isBusy;
  });
}

function setStatus(text) {
  $("status").textContent = text;
}

function updateLabels() {
  $("blockValue").textContent = $("blockSize").value;
  $("uniqueValue").textContent = $("uniqueness").value;
  $("speckleWindowValue").textContent = $("speckleWindow").value;
  $("speckleRangeValue").textContent = $("speckleRange").value;
  $("minDispValue").textContent = $("minDisparity").value;
  $("numDispValue").textContent = $("numDisparities").value;
}

function applyParams(values) {
  $("algorithm").value = values.algorithm;
  $("blockSize").value = values.block_size;
  $("uniqueness").value = values.uniqueness_ratio;
  $("speckleWindow").value = values.speckle_window_size;
  $("speckleRange").value = values.speckle_range;
  $("minDisparity").value = values.min_disparity;
  $("numDisparities").value = values.num_disparities;
  updateLabels();
}

function renderTabs() {
  const tabs = $("tabs");
  tabs.innerHTML = "";
  views.forEach((view) => {
    const button = document.createElement("button");
    button.textContent = view.replace("_", " ");
    button.className = view === currentView ? "active" : "";
    button.onclick = () => {
      currentView = view;
      renderTabs();
      renderImage();
    };
    tabs.appendChild(button);
  });
}

function renderImage() {
  if (!lastPayload) {
    return;
  }
  $("viewImage").src = `data:image/png;base64,${lastPayload.images[currentView]}`;
}

function renderMetrics(metrics) {
  $("mae").textContent = `${metrics.mae.toFixed(2)} px`;
  $("bad3").textContent = `${metrics.bad3.toFixed(2)}%`;
}

function applyResult(payload) {
  lastPayload = payload;
  renderMetrics(payload.metrics);
  renderImage();
  setStatus(payload.status);
}

async function postJson(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return await response.json();
}

async function compute() {
  setBusy(true);
  setStatus("Computing disparity...");
  try {
    const payload = await postJson("/api/compute", params());
    applyResult(payload);
  } catch (error) {
    setStatus(`Error: ${error.message}`);
  } finally {
    setBusy(false);
  }
}

async function save() {
  setBusy(true);
  setStatus("Exporting current result...");
  try {
    const payload = await postJson("/api/save", params());
    setStatus(payload.status);
  } catch (error) {
    setStatus(`Error: ${error.message}`);
  } finally {
    setBusy(false);
  }
}

async function saveBenchmark() {
  if (!lastPayload) {
    await compute();
  }
  setStatus("Saving test...");
  try {
    const payload = await postJson("/api/test", params());
    tests = payload.tests;
    renderBenchmarks();
    setStatus(payload.status);
  } catch (error) {
    setStatus(`Error: ${error.message}`);
  }
}

function compareBenchmarks(a, b, metric) {
  if (a.metrics[metric] !== b.metrics[metric]) {
    return a.metrics[metric] - b.metrics[metric];
  }
  if (a.metrics.mae !== b.metrics.mae) {
    return a.metrics.mae - b.metrics.mae;
  }
  return a.metrics.bad3 - b.metrics.bad3;
}

function findTestById(testId) {
  return tests.find((item) => Number(item.id) === Number(testId));
}

async function loadBenchmarks() {
  try {
    const response = await fetch("/api/tests");
    const payload = await response.json();
    tests = payload.tests || [];
    renderBenchmarks();
  } catch (error) {
    setStatus(`Could not load tests: ${error.message}`);
  }
}

function benchmarkTable(rows) {
  let html = "<table><thead><tr><th>ID</th><th>Block</th><th>Uniq.</th><th>Disp</th><th>Speckle</th><th>MAE</th><th>Bad3</th></tr></thead><tbody>";
  rows.forEach((item, index) => {
    const p = item.params;
    const m = item.metrics;
    html += `<tr class="${index === 0 ? "best" : ""}"><td>${item.id ?? "-"}</td><td>${p.block_size}</td><td>${p.uniqueness_ratio}</td><td>${p.min_disparity}/${p.num_disparities}</td><td>${p.speckle_window_size}/${p.speckle_range}</td><td>${m.mae.toFixed(2)}</td><td>${m.bad3.toFixed(2)}%</td></tr>`;
  });
  html += "</tbody></table>";
  return html;
}

function renderBenchmarks() {
  const target = $("benchmarks");
  if (!tests.length) {
    target.innerHTML = '<div class="benchmark-grid"><section class="algorithm-panel"><div class="status">No saved tests yet.</div></section></div>';
    return;
  }

  let html = '<div class="benchmark-grid">';
  ["sgbm", "bm"].forEach((algorithm) => {
    const rows = tests
      .filter((item) => item.params.algorithm === algorithm)
      .sort((a, b) => compareBenchmarks(a, b, benchmarkSortMetric));
    const best = rows[0];
    html += `<section class="algorithm-panel"><div class="algorithm-title">${algorithm.toUpperCase()} Tests (${rows.length})</div>`;
    if (best) {
      html += `<div class="sub">MAE ${best.metrics.mae.toFixed(2)} px | bad ${best.metrics.bad3.toFixed(2)}%</div>`;
      html += benchmarkTable(rows);
    } else {
      html += `<div class="status">No ${algorithm.toUpperCase()} tests yet.</div>`;
    }
    html += "</section>";
  });
  html += "</div>";
  target.innerHTML = html;
}

async function loadTestById() {
  const testId = Number($("testId").value);
  if (!Number.isInteger(testId) || testId < 1) {
    setStatus("Please enter a valid test ID.");
    return;
  }
  const test = findTestById(testId);
  if (!test) {
    setStatus(`No saved test with ID ${testId}.`);
    return;
  }
  applyParams(test.params);
  setStatus(`Loaded test #${test.id}: ${test.params.algorithm.toUpperCase()}, MAE ${test.metrics.mae.toFixed(2)} px.`);
}

["blockSize", "uniqueness", "speckleWindow", "speckleRange", "minDisparity", "numDisparities"].forEach((id) => {
  $(id).addEventListener("input", updateLabels);
});

$("sortMetric").addEventListener("change", (event) => {
  benchmarkSortMetric = event.target.value;
  renderBenchmarks();
});

$("compute").onclick = compute;
$("save").onclick = save;
$("benchmark").onclick = saveBenchmark;
$("loadTest").onclick = loadTestById;
$("testId").addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    loadTestById();
  }
});

renderTabs();
updateLabels();
loadBenchmarks();
compute();
