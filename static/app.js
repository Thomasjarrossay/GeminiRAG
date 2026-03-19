/* ═══════════════════════════════════════════════════════════════
   GeminiRAG — Frontend App
   ═══════════════════════════════════════════════════════════════ */

const API = {
  ingest:    "/api/ingest",
  query:     "/api/query",
  documents: "/api/documents",
  health:    "/api/health",
  stats:     "/api/stats",
};

// ─── State ────────────────────────────────────────────────────────
let currentSources = [];
let isLoading = false;

// ─── DOM Refs ─────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const msgContainer    = $("messages");
const questionInput   = $("question-input");
const sendBtn         = $("send-btn");
const docList         = $("doc-list");
const statsCount      = $("stats-count");
const sourcesList     = $("sources-list");
const sourcesPanel    = $("sources-panel");
const uploadProgress  = $("upload-progress");
const progressBar     = $("progress-bar");
const progressLabel   = $("progress-label");
const typeFilter      = $("type-filter");

// ─── Init ─────────────────────────────────────────────────────────
async function init() {
  setupDropZones();
  setupInput();
  setupButtons();
  await Promise.all([loadDocuments(), loadStats()]);
}

// ─── Drop Zones ───────────────────────────────────────────────────
function setupDropZones() {
  const zones = [
    { zone: $("drop-videos"), input: $("input-videos") },
    { zone: $("drop-images"), input: $("input-images") },
    { zone: $("drop-texts"),  input: $("input-texts")  },
  ];

  zones.forEach(({ zone, input }) => {
    // Click to browse
    zone.addEventListener("click", () => input.click());

    // Drag & drop
    zone.addEventListener("dragover", (e) => {
      e.preventDefault();
      zone.classList.add("drag-over");
    });
    zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      zone.classList.remove("drag-over");
      const files = Array.from(e.dataTransfer.files);
      if (files.length) uploadFiles(files);
    });

    // File input
    input.addEventListener("change", (e) => {
      const files = Array.from(e.target.files);
      if (files.length) uploadFiles(files);
      input.value = "";
    });
  });
}

// ─── Upload Files ─────────────────────────────────────────────────
async function uploadFiles(files) {
  showProgress(0, `Indexation de ${files.length} fichier${files.length > 1 ? "s" : ""}…`);

  let successCount = 0;
  let errorCount = 0;

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const pct = Math.round(((i) / files.length) * 100);
    showProgress(pct, `Indexation : ${file.name}…`);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(API.ingest, { method: "POST", body: formData });
      const data = await res.json();

      if (res.ok) {
        successCount++;
        toast(`✓ ${file.name} — ${data.chunks_indexed} chunks indexés`, "success");
      } else {
        errorCount++;
        toast(`✗ ${file.name} : ${data.detail || "Erreur d'indexation"}`, "error");
      }
    } catch (e) {
      errorCount++;
      toast(`✗ ${file.name} : Erreur réseau`, "error");
    }
  }

  showProgress(100, "Terminé");
  setTimeout(() => hideProgress(), 1200);

  await Promise.all([loadDocuments(), loadStats()]);
}

function showProgress(pct, label) {
  uploadProgress.style.display = "block";
  progressBar.style.width = pct + "%";
  progressLabel.textContent = label;
}

function hideProgress() {
  uploadProgress.style.display = "none";
  progressBar.style.width = "0%";
}

// ─── Documents List ───────────────────────────────────────────────
async function loadDocuments() {
  try {
    const res = await fetch(API.documents);
    const data = await res.json();
    renderDocuments(data.documents || []);
  } catch (e) {
    console.warn("Erreur chargement documents:", e);
  }
}

function renderDocuments(docs) {
  if (!docs.length) {
    docList.innerHTML = `
      <div class="empty-state">
        <p>Aucun document indexé</p>
        <p class="empty-sub">Déposez des fichiers ci-dessus</p>
      </div>`;
    return;
  }

  docList.innerHTML = docs.map(doc => `
    <div class="doc-item" data-file="${escHtml(doc.source_file)}">
      <div class="doc-info">
        <div class="doc-name" title="${escHtml(doc.source_file)}">${escHtml(doc.source_file)}</div>
        <div class="doc-meta">
          <span class="type-badge ${doc.file_type}">${typeIcon(doc.file_type)} ${doc.file_type}</span>
          <span>${doc.chunk_count} chunk${doc.chunk_count > 1 ? "s" : ""}</span>
        </div>
      </div>
      <button class="doc-delete" title="Supprimer" onclick="deleteDocument('${escHtml(doc.source_file)}')">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
    </div>
  `).join("");
}

async function deleteDocument(sourceFile) {
  if (!confirm(`Supprimer "${sourceFile}" de la base ?`)) return;
  try {
    const res = await fetch(API.documents, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_file: sourceFile }),
    });
    if (res.ok) {
      toast(`"${sourceFile}" supprimé`, "info");
      await Promise.all([loadDocuments(), loadStats()]);
    }
  } catch (e) {
    toast("Erreur lors de la suppression", "error");
  }
}

// ─── Stats ────────────────────────────────────────────────────────
async function loadStats() {
  try {
    const res = await fetch(API.stats);
    const data = await res.json();
    statsCount.textContent = `${data.total_vectors} vecteurs`;
  } catch (e) {
    statsCount.textContent = "— vecteurs";
  }
}

// ─── Chat ─────────────────────────────────────────────────────────
function setupInput() {
  // Auto-resize textarea
  questionInput.addEventListener("input", () => {
    questionInput.style.height = "auto";
    questionInput.style.height = Math.min(questionInput.scrollHeight, 160) + "px";
  });

  // Ctrl+Enter to send
  questionInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      sendMessage();
    }
  });
}

function setupButtons() {
  sendBtn.addEventListener("click", sendMessage);
  $("clear-chat").addEventListener("click", clearChat);
  $("refresh-docs").addEventListener("click", async () => {
    await Promise.all([loadDocuments(), loadStats()]);
    toast("Base actualisée", "info");
  });
  $("close-sources").addEventListener("click", closeSources);
}

async function sendMessage() {
  const question = questionInput.value.trim();
  if (!question || isLoading) return;

  isLoading = true;
  sendBtn.disabled = true;

  // Masquer welcome card
  const welcome = document.querySelector(".welcome-card");
  if (welcome) welcome.remove();

  // Message utilisateur
  appendMessage("user", question);
  questionInput.value = "";
  questionInput.style.height = "auto";

  // Indicateur de réflexion
  const thinkingId = appendThinking();

  try {
    const res = await fetch(API.query, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        top_k: 5,
        file_type_filter: typeFilter.value || null,
      }),
    });

    removeThinking(thinkingId);

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      appendMessage("assistant", `❌ Erreur : ${err.detail || res.statusText}`, [], 0);
      return;
    }

    const data = await res.json();
    currentSources = data.sources || [];
    appendMessage("assistant", data.answer, currentSources, data.tokens_used);

  } catch (e) {
    removeThinking(thinkingId);
    appendMessage("assistant", `❌ Erreur réseau : ${e.message}`, [], 0);
  } finally {
    isLoading = false;
    sendBtn.disabled = false;
    questionInput.focus();
  }
}

function appendMessage(role, content, sources = [], tokens = 0) {
  const id = "msg-" + Date.now();
  const initials = role === "user" ? "T" : "◈";

  const sourcesBtn = sources.length
    ? `<button class="sources-btn" onclick="openSources(${id})">
         <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
           <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
         </svg>
         ${sources.length} source${sources.length > 1 ? "s" : ""}
       </button>`
    : "";

  const tokenStr = tokens
    ? `<span class="token-count">${tokens} tokens</span>`
    : "";

  const footer = (sourcesBtn || tokenStr)
    ? `<div class="bubble-footer">${sourcesBtn}${tokenStr}</div>`
    : "";

  const html = `
    <div class="message ${role}" id="${id}" data-sources='${JSON.stringify(sources)}'>
      <div class="avatar">${initials}</div>
      <div class="bubble">
        ${renderMarkdown(content)}
        ${footer}
      </div>
    </div>`;

  msgContainer.insertAdjacentHTML("beforeend", html);
  scrollToBottom();
}

function appendThinking() {
  const id = "thinking-" + Date.now();
  const html = `
    <div class="message assistant" id="${id}">
      <div class="avatar">◈</div>
      <div class="bubble">
        <div class="thinking">
          <div class="thinking-dots">
            <span></span><span></span><span></span>
          </div>
          Recherche dans la base…
        </div>
      </div>
    </div>`;
  msgContainer.insertAdjacentHTML("beforeend", html);
  scrollToBottom();
  return id;
}

function removeThinking(id) {
  const el = $(id);
  if (el) el.remove();
}

function clearChat() {
  msgContainer.innerHTML = `
    <div class="welcome-card">
      <div class="welcome-icon">◈</div>
      <h2>Bienvenue sur GeminiRAG</h2>
      <p>Indexez vos vidéos, images et documents, puis posez vos questions.<br>
      L'IA interroge votre base de connaissances multimodale.</p>
      <div class="welcome-chips">
        <button class="chip" onclick="insertExample('Quels sont les thèmes principaux dans les documents indexés ?')">Thèmes principaux</button>
        <button class="chip" onclick="insertExample('Résume les vidéos disponibles')">Résumé vidéos</button>
        <button class="chip" onclick="insertExample('Quelles images ont été indexées ?')">Images indexées</button>
      </div>
    </div>`;
  closeSources();
}

// ─── Sources Panel ────────────────────────────────────────────────
function openSources(msgId) {
  const msgEl = typeof msgId === "string" ? $(msgId) : msgId;
  if (!msgEl) return;

  let sources;
  try {
    sources = JSON.parse(msgEl.dataset.sources || "[]");
  } catch {
    sources = [];
  }

  if (!sources.length) return;

  sourcesList.innerHTML = sources.map((s, i) => `
    <div class="source-card">
      <div class="source-card-header">
        <span class="source-file">${typeIcon(s.file_type)} ${escHtml(s.source_file)}</span>
        <span class="source-score">${(s.score * 100).toFixed(1)}%</span>
      </div>
      <span class="type-badge ${s.file_type}">${s.file_type}</span>
      <p class="source-content" style="margin-top:6px">${escHtml(s.content)}</p>
    </div>
  `).join("");

  document.body.classList.add("sources-open");
}

function closeSources() {
  document.body.classList.remove("sources-open");
}

// ─── Utils ────────────────────────────────────────────────────────
function scrollToBottom() {
  msgContainer.scrollTop = msgContainer.scrollHeight;
}

function insertExample(text) {
  questionInput.value = text;
  questionInput.focus();
  questionInput.dispatchEvent(new Event("input"));
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function typeIcon(type) {
  return { text: "📄", image: "🖼️", video: "🎬", pdf: "📋" }[type] || "📎";
}

function renderMarkdown(text) {
  // Minimal markdown : code blocks, inline code, bold, italic, lists
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    // Code blocks
    .replace(/```[\w]*\n?([\s\S]*?)```/g, "<pre><code>$1</code></pre>")
    // Inline code
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    // Bold
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    // Italic
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    // Links [text](url)
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener" style="color:#818CF8">$1</a>')
    // Unordered lists
    .replace(/^- (.+)$/gm, "<li>$1</li>")
    .replace(/(<li>.*<\/li>\n?)+/g, "<ul>$&</ul>")
    // Line breaks
    .replace(/\n{2,}/g, "</p><p>")
    .replace(/\n/g, "<br>")
    // Wrap in paragraphs
    .replace(/^(?!<[a-z])(.+)/gm, (m) => m.startsWith("<") ? m : `<p>${m}</p>`);
}

function toast(message, type = "info") {
  const container = $("toast-container");
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = message;
  container.appendChild(el);
  setTimeout(() => el.remove(), 3500);
}

// ─── Start ────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", init);
