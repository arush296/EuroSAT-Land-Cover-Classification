const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1"]);

window.EUROSAT_API_URL = LOCAL_HOSTS.has(window.location.hostname)
  ? "http://127.0.0.1:8000"
  : "https://eurosat-api.onrender.com";
