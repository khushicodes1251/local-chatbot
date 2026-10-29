const state = {
  files: [],
  selected: [],
  messages: [],
  busy: false,
  indexing: false,
};

const els = {
  dropzone: document.getElementById("dropzone"),
  dropTitle: document.getElementById("drop-title"),
  fileInput: document.getElementById("file-input"),
  notice: document.getElementById("notice"),
  selectedCount: document.getElementById("selected-count"),
  empty: document.getElementById("empty"),
  library: document.getElementById("library"),
  welcome: document.getElementById("welcome"),
  transcript: document.getElementById("transcript"),
  scope: document.getElementById("scope"),
  question: document.getElementById("question"),
  send: document.getElementById("send"),
  composer: document.getElementById("composer"),
};

function selectedReady() {
  return state.files.filter(
    (file) => state.selected.includes(file.name) && file.status === "ready"
  );
}

function setStatus(text, error) {
  els.notice.textContent = text || "";
  els.notice.classList.toggle("error", Boolean(error));
}

function renderLibrary() {
  els.selectedCount.textContent = `${state.selected.length} selected`;
  els.empty.hidden = state.files.length > 0;
  els.library.hidden = state.files.length === 0;
  els.library.replaceChildren();

  for (const file of state.files) {
    const item = document.createElement("li");
    if (state.selected.includes(file.name)) item.classList.add("selected");

    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = state.selected.includes(file.name);
    checkbox.addEventListener("change", () => toggle(file.name));

    const text = document.createElement("span");
    const name = document.createElement("strong");
    name.textContent = file.name;
    name.title = file.name;
    const meta = document.createElement("small");
    const pages = file.pages
      ? ` · ${file.pages} page${file.pages === 1 ? "" : "s"}`
      : "";
    meta.textContent = `${file.status || "ready"}${pages}`;
    text.append(name, meta);
    label.append(checkbox, text);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "text-button";
    remove.textContent = "Remove";
    remove.addEventListener("click", () => onRemove(file.name));

    item.append(label, remove);
    els.library.append(item);
  }
}

function renderComposer() {
  const names = selectedReady().map((file) => file.name);
  els.scope.textContent = names.length
    ? `Asking ${names.join(", ")}`
    : "Select a ready PDF to enable questions.";
  const canAsk = names.length > 0 && !state.busy;
  els.question.disabled = !canAsk;
  els.question.placeholder = names.length
    ? "Ask about the selected PDFs…"
    : "Select a PDF first";
  els.send.disabled = !canAsk || !els.question.value.trim();
  els.send.textContent = state.busy ? "Sending" : "Send";
  els.dropzone.classList.toggle("busy", state.indexing);
  els.fileInput.disabled = state.indexing;
  els.dropTitle.textContent = state.indexing ? "Indexing PDF…" : "Drop a PDF here";
}

function renderEvidence(source) {
  const item = document.createElement("li");
  const cite = document.createElement("p");
  cite.className = "source-cite";
  cite.textContent = source.page ? `${source.name}, page ${source.page}` : source.name;
  item.append(cite);

  const text = source.text || "";
  if (!text) return item;

  const quote = document.createElement("blockquote");
  const long = text.length > 180;
  quote.className = long ? "excerpt" : "excerpt open";
  quote.textContent = text;
  item.append(quote);

  if (long) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "text-button";
    button.textContent = "Show more";
    button.addEventListener("click", () => {
      const open = quote.classList.toggle("open");
      button.textContent = open ? "Show less" : "Show more";
    });
    item.append(button);
  }
  return item;
}

function renderBubble(message, index) {
  const article = document.createElement("article");
  article.className = `bubble ${message.role}`;
  const paragraph = document.createElement("p");
  const waiting = state.busy && index === state.messages.length - 1 && !message.text;
  paragraph.textContent = message.text || (waiting ? "Reading the selected pages…" : "");
  article.append(paragraph);

  if (message.sources && message.sources.length) {
    const list = document.createElement("ul");
    list.className = "sources";
    for (const source of message.sources) list.append(renderEvidence(source));
    article.append(list);
  }
  return article;
}

function renderTranscript() {
  const hasMessages = state.messages.length > 0;
  els.welcome.hidden = hasMessages;
  els.transcript.hidden = !hasMessages;
  els.transcript.replaceChildren();
  state.messages.forEach((message, index) => {
    els.transcript.append(renderBubble(message, index));
  });
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

function render() {
  renderLibrary();
  renderComposer();
  renderTranscript();
}

function toggle(name) {
  state.selected = state.selected.includes(name)
    ? state.selected.filter((item) => item !== name)
    : state.selected.concat(name);
  renderLibrary();
  renderComposer();
}

async function readError(response, fallback) {
  try {
    const data = await response.json();
    if (data && data.error) return data.error;
  } catch (_error) {
    /* The server sometimes returns an empty body. */
  }
  return fallback;
}

async function uploadPdf(file) {
  const body = new FormData();
  body.append("file", file);
  const response = await fetch("/api/upload", { method: "POST", body });
  if (!response.ok) throw new Error(await readError(response, "Upload failed."));
  return response.json();
}

async function removePdf(name) {
  const response = await fetch(`/api/files/${encodeURIComponent(name)}`, { method: "DELETE" });
  if (!response.ok) throw new Error(await readError(response, "Could not remove that document."));
  const data = await response.json();
  return data.files || [];
}

async function streamAnswer(question, documents, onEvent, history) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, documents, history }),
  });
  if (!response.ok || !response.body) {
    throw new Error(await readError(response, "The chat request failed."));
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) {
      const line = part.split("\n").find((row) => row.startsWith("data: "));
      if (!line) continue;
      onEvent(JSON.parse(line.slice(6)));
    }
  }
}

async function onUpload(file) {
  if (!file || state.indexing) return;
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    setStatus("Upload a PDF. Other file types are not accepted.", true);
    return;
  }
  state.indexing = true;
  setStatus(`Indexing ${file.name}…`, false);
  renderComposer();
  try {
    const data = await uploadPdf(file);
    state.files = data.files || [];
    if (!state.selected.includes(data.name)) state.selected.push(data.name);
    setStatus(
      data.duplicate
        ? `${data.name} is already indexed and selected.`
        : `${data.name} is ready. It is selected for questions.`,
      false
    );
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    state.indexing = false;
    render();
  }
}

async function onRemove(name) {
  try {
    state.files = await removePdf(name);
    state.selected = state.selected.filter((item) => item !== name);
    setStatus(`Removed ${name}.`, false);
    render();
  } catch (error) {
    setStatus(error.message, true);
  }
}

function applyEvent(event) {
  const last = state.messages[state.messages.length - 1];
  if (!last) return;
  if (event.token) last.text += event.token;
  if (event.error) last.text += event.error;
  if (event.sources) last.sources = event.sources;
  renderTranscript();
  renderComposer();
}

async function onAsk(question) {
  const documents = selectedReady().map((file) => file.name);
  if (!documents.length) {
    setStatus("Select at least one indexed PDF before asking.", true);
    return;
  }
  const history = state.messages
    .slice(-4)
    .map((message) => ({ role: message.role, text: (message.text || "").slice(0, 500) }))
    .filter((message) => message.text);

  state.busy = true;
  state.messages.push(
    { role: "user", text: question },
    { role: "assistant", text: "", sources: [] }
  );
  render();
  try {
    await streamAnswer(question, documents, applyEvent, history);
  } catch (error) {
    const last = state.messages[state.messages.length - 1];
    if (last) last.text = error.message;
  } finally {
    state.busy = false;
    render();
  }
}

els.fileInput.addEventListener("change", () => {
  const file = els.fileInput.files && els.fileInput.files[0];
  els.fileInput.value = "";
  onUpload(file);
});

["dragenter", "dragover"].forEach((name) => {
  els.dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    els.dropzone.classList.add("over");
  });
});
els.dropzone.addEventListener("dragleave", () => els.dropzone.classList.remove("over"));
els.dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  els.dropzone.classList.remove("over");
  const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
  onUpload(file);
});

els.question.addEventListener("input", renderComposer);
els.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = els.question.value.trim();
  if (!question || state.busy || els.question.disabled) return;
  els.question.value = "";
  onAsk(question);
});

fetch("/api/files")
  .then(async (response) => {
    if (!response.ok) throw new Error(await readError(response, "Could not load documents."));
    return response.json();
  })
  .then((data) => {
    state.files = data.files || [];
    render();
  })
  .catch((error) => setStatus(error.message, true));
