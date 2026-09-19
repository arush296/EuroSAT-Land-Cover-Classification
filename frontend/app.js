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

let previewUrl;
let backendRetryTimer;

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

input.addEventListener("change", () => {
  clearResults();

  const file = input.files[0];
  button.disabled = !file;
  previewWrapper.hidden = !file;

  if (!file) return;

  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  fileName.textContent = file.name;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = input.files[0];
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
