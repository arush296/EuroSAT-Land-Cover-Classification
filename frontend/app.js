const API_URL = window.EUROSAT_API_URL.replace(/\/$/, "");

const form = document.querySelector("#prediction-form");
const input = document.querySelector("#image-input");
const button = document.querySelector("#predict-button");
const previewWrapper = document.querySelector("#preview-wrapper");
const preview = document.querySelector("#image-preview");
const fileName = document.querySelector("#file-name");
const status = document.querySelector("#status");
const results = document.querySelector("#results");
const predictedClass = document.querySelector("#predicted-class");
const predictedScore = document.querySelector("#predicted-score");
const rankings = document.querySelector("#top-predictions");
const backendStatus = document.querySelector("#backend-status");
const exampleGrid = document.querySelector("#example-grid");
const shuffleExamplesButton = document.querySelector("#shuffle-examples");

let previewUrl;
let backendRetryTimer;
let selectedFile;
let selectedExamplePath;
let examplePool = [];

function setBackendConnected() {
  if (backendRetryTimer) clearTimeout(backendRetryTimer);
  backendRetryTimer = undefined;
  backendStatus.textContent = "Backend connected";
  backendStatus.classList.add("connected");
}

function setBackendUnavailable() {
  backendStatus.textContent = "Backend unavailable — retrying…";
  backendStatus.classList.remove("connected");

  if (!backendRetryTimer) {
    backendRetryTimer = setTimeout(() => {
      backendRetryTimer = undefined;
      checkBackend();
    }, 5000);
  }
}

function percentage(score) {
  return `${(score * 100).toFixed(1)}%`;
}

function clearResults() {
  results.hidden = true;
  rankings.replaceChildren();
  status.textContent = "";
}

function formatClassName(className) {
  return className.replace(/([a-z])([A-Z])/g, "$1 $2");
}

function markSelectedExample() {
  for (const card of exampleGrid.querySelectorAll(".example-card")) {
    card.classList.toggle("selected", card.dataset.file === selectedExamplePath);
  }
}

function selectFile(file, displayName, examplePath) {
  clearResults();
  selectedFile = file;
  selectedExamplePath = examplePath;
  button.disabled = !file;
  previewWrapper.hidden = !file;

  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = file ? URL.createObjectURL(file) : undefined;
  preview.removeAttribute("src");

  if (file) {
    preview.src = previewUrl;
    fileName.textContent = displayName;
  } else {
    fileName.textContent = "";
  }

  markSelectedExample();
}

input.addEventListener("change", () => {
  const file = input.files[0];
  if (file) selectFile(file, file.name);
});

async function selectExample(example) {
  status.textContent = "Loading example…";

  try {
    const response = await fetch(example.file);
    if (!response.ok) throw new Error();

    const blob = await response.blob();
    const filename = example.file.split("/").pop();
    const file = new File([blob], filename, {
      type: blob.type || "image/jpeg",
    });

    input.value = "";
    selectFile(
      file,
      `${formatClassName(example.class_name)} example`,
      example.file,
    );
  } catch {
    status.textContent = "Could not load that example. Please try another.";
  }
}

function randomExamplesByClass() {
  const grouped = new Map();

  for (const example of examplePool) {
    if (!grouped.has(example.class_name)) grouped.set(example.class_name, []);
    grouped.get(example.class_name).push(example);
  }

  return [...grouped.values()].map(
    (examples) => examples[Math.floor(Math.random() * examples.length)],
  );
}

function renderExamples() {
  exampleGrid.replaceChildren();

  for (const example of randomExamplesByClass()) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "example-card";
    card.dataset.file = example.file;
    card.setAttribute(
      "aria-label",
      `Use ${formatClassName(example.class_name)} example`,
    );

    const image = document.createElement("img");
    image.src = example.file;
    image.alt = "";
    image.loading = "lazy";

    const label = document.createElement("span");
    label.textContent = formatClassName(example.class_name);

    card.append(image, label);
    card.addEventListener("click", () => selectExample(example));
    exampleGrid.append(card);
  }

  markSelectedExample();
}

async function loadExamples() {
  try {
    const response = await fetch("examples/manifest.json");
    if (!response.ok) throw new Error();

    const manifest = await response.json();
    examplePool = manifest.examples;
    shuffleExamplesButton.disabled = false;
    renderExamples();
  } catch {
    exampleGrid.innerHTML =
      '<p class="examples-loading">Examples unavailable. Upload your own image instead.</p>';
    shuffleExamplesButton.disabled = true;
  }
}

shuffleExamplesButton.addEventListener("click", renderExamples);

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = selectedFile;
  if (!file) return;

  clearResults();
  button.disabled = true;
  button.textContent = "Classifying…";
  status.textContent = "Uploading image…";

  const formData = new FormData();
  formData.append("file", file);
  let backendReached = false;

  try {
    const response = await fetch(`${API_URL}/predict?top_k=3`, {
      method: "POST",
      body: formData,
    });
    backendReached = true;
    setBackendConnected();
    const body = await response.json();

    if (!response.ok) {
      throw new Error(body.detail || "Prediction failed.");
    }

    predictedClass.textContent = body.predicted_class;
    predictedScore.textContent = percentage(body.score);

    for (const prediction of body.top_predictions) {
      const row = document.createElement("div");
      row.className = "ranking-row";
      row.innerHTML = `
        <span>${prediction.class_name}</span>
        <strong>${percentage(prediction.score)}</strong>
        <div class="ranking-bar" aria-hidden="true">
          <span style="width: ${prediction.score * 100}%"></span>
        </div>
      `;
      rankings.append(row);
    }

    status.textContent = "";
    results.hidden = false;
  } catch (error) {
    status.textContent = error.message || "Could not reach the prediction API.";
    if (!backendReached) setBackendUnavailable();
  } finally {
    button.disabled = false;
    button.textContent = "Classify image";
  }
});

async function checkBackend() {
  try {
    const response = await fetch(`${API_URL}/health`);
    if (!response.ok) throw new Error();

    setBackendConnected();
  } catch {
    setBackendUnavailable();
  }
}

checkBackend();
loadExamples();
