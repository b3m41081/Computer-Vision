const ids = (name) => document.getElementById(name);
let lastStatus = null;
let wasRunning = false;

function parameters(action) {
  return {
    action,
    video: ids("video").value,
    interval: Number(ids("interval").value),
    max_frames: Number(ids("maxFrames").value),
    min_blur_score: Number(ids("blur").value),
    max_size: Number(ids("maxSize").value),
    max_features: Number(ids("features").value),
    sequential_overlap: Number(ids("overlap").value),
    da3_device: ids("da3Device").value,
    da3_model: ids("da3Model").value,
    da3_max_images: Number(ids("da3Images").value),
    da3_max_points: Number(ids("da3Points").value),
    da3_confidence: Number(ids("da3Confidence").value),
    da3_resolution: Number(ids("da3Resolution").value),
  };
}

async function post(path, body = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Request failed");
  return payload;
}

function number(value, digits = 0) {
  if (value === undefined || value === null) return "-";
  return Number(value).toLocaleString("de-DE", { maximumFractionDigits: digits });
}

function updateVideos(videos) {
  const select = ids("video");
  const selected = select.value;
  const names = videos.map((item) => item.name);
  if (Array.from(select.options).map((option) => option.value).join("|") === names.join("|")) return;
  select.innerHTML = "";
  videos.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.name;
    option.textContent = `${item.name} (${(item.size / 1024 / 1024).toFixed(1)} MB)`;
    select.appendChild(option);
  });
  if (names.includes(selected)) select.value = selected;
}

function setBusy(running) {
  ["runFull", "runExtract", "runColmap", "runDa3", "runVisualize"].forEach((id) => {
    ids(id).disabled = running;
  });
  ids("runVisualizeDa3").disabled = running || !lastStatus?.results?.da3_model_exists;
  ids("cancel").disabled = !running;
}

function updateResults(results = {}) {
  ids("frames").textContent = number(results.exported_frames ?? results.image_count);
  ids("registered").textContent = number(results.registered_images);
  ids("points").textContent = number(results.points);
  ids("da3ImagesMetric").textContent = number(results.da3_images);
  ids("da3PointsMetric").textContent = number(results.da3_points);
  ids("da3DeviceMetric").textContent = results.da3_device || "-";
  ids("observations").textContent = number(results.observations);
  ids("track").textContent = number(results.mean_track_length, 2);
  ids("error").textContent = results.mean_reprojection_error == null
    ? "-" : `${number(results.mean_reprojection_error, 2)} px`;

  ids("modelLink").classList.toggle("disabled", !results.model_exists);
  ids("imageLink").classList.toggle("disabled", !results.screenshot_exists);
  ids("manifestLink").classList.toggle("disabled", !results.exported_frames);
  ids("da3ModelLink").classList.toggle("disabled", !results.da3_model_exists);
  ids("da3ImageLink").classList.toggle("disabled", !results.da3_screenshot_exists);
  ids("da3MetadataLink").classList.toggle("disabled", !results.da3_points);
  if (results.screenshot_exists) {
    ids("resultImage").src = `/result/screenshot?v=${results.screenshot_version}`;
    ids("resultImage").classList.add("visible");
    ids("imageEmpty").classList.add("hidden");
  }
  if (results.da3_screenshot_exists) {
    ids("da3ResultImage").src = `/result/da3-screenshot?v=${results.da3_screenshot_version}`;
    ids("da3ResultImage").classList.add("visible");
    ids("da3ImageEmpty").classList.add("hidden");
  }
}

function renderStatus(payload) {
  lastStatus = payload;
  const running = payload.status === "running";
  ids("stage").textContent = payload.stage || "Bereit";
  ids("badge").textContent = payload.status;
  ids("badge").className = `badge ${payload.status}`;
  setBusy(running);
  updateVideos(payload.videos || []);
  updateResults(payload.results || {});

  const log = ids("log");
  const text = (payload.log || []).join("\n") || (payload.error ? payload.error : "Bereit.");
  if (log.textContent !== text) {
    log.textContent = text;
    log.scrollTop = log.scrollHeight;
  }
  ids("timing").textContent = payload.started_at
    ? `${payload.started_at}${payload.finished_at ? ` bis ${payload.finished_at}` : " · läuft"}`
    : "Noch kein Lauf gestartet.";

  if (wasRunning && !running && payload.status === "success") {
    refreshStatus();
  }
  wasRunning = running;
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    renderStatus(await response.json());
  } catch (error) {
    ids("stage").textContent = `Verbindung fehlgeschlagen: ${error.message}`;
  }
}

async function run(action) {
  try {
    renderStatus(await post("/api/run", parameters(action)));
  } catch (error) {
    ids("stage").textContent = `Fehler: ${error.message}`;
  }
}

ids("runFull").onclick = () => run("full");
ids("runExtract").onclick = () => run("extract");
ids("runColmap").onclick = () => run("reconstruct");
ids("runDa3").onclick = () => run("da3");
ids("runVisualize").onclick = () => run("visualize");
ids("runVisualizeDa3").onclick = () => run("visualize_da3");
ids("cancel").onclick = async () => renderStatus(await post("/api/cancel"));
ids("refreshImage").onclick = () => {
  if (lastStatus?.results?.screenshot_exists) {
    ids("resultImage").src = `/result/screenshot?v=${Date.now()}`;
  }
};
ids("refreshDa3Image").onclick = () => {
  if (lastStatus?.results?.da3_screenshot_exists) {
    ids("da3ResultImage").src = `/result/da3-screenshot?v=${Date.now()}`;
  }
};

refreshStatus();
setInterval(refreshStatus, 1000);
