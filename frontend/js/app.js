// ── Config ────────────────────────────────────────────────────────────────
const API_BASE = 'http://localhost:8000';

// ── State ─────────────────────────────────────────────────────────────────
let threadId = null;
let isLoading = false;
let isInterrupted = false;

// ── DOM refs ──────────────────────────────────────────────────────────────
const messagesEl  = document.getElementById('messages');
const inputEl     = document.getElementById('input');
const sendBtn     = document.getElementById('sendBtn');
const statusDot   = document.getElementById('statusDot');
const statusLabel = document.getElementById('statusLabel');
const threadDisp  = document.getElementById('threadDisplay');
const emptyState  = document.getElementById('emptyState');

// ── Health check ──────────────────────────────────────────────────────────
// Pings the backend every 15s to show online/offline status in the header
async function checkHealth() {
  try {
    const r = await fetch(`${API_BASE}/health`);
    const d = await r.json();
    if (d.agent_ready) {
      statusDot.classList.add('online');
      statusLabel.textContent = 'online';
    } else {
      statusLabel.textContent = 'starting';
    }
  } catch {
    statusDot.classList.remove('online');
    statusLabel.textContent = 'offline';
  }
}
checkHealth();
setInterval(checkHealth, 15000);

// ── Textarea auto-resize ──────────────────────────────────────────────────
// Grows the textarea as the user types, up to a max height
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 140) + 'px';
}

// ── Key handler ───────────────────────────────────────────────────────────
// Enter sends, Shift+Enter adds a new line
function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

// ── Suggestion fill ───────────────────────────────────────────────────────
// Clicking a suggestion card fills the input with its text
function fillSuggestion(el) {
  inputEl.value = el.querySelector('.suggestion-text').textContent;
  autoResize(inputEl);
  inputEl.focus();
}

// ── Clear thread ──────────────────────────────────────────────────────────
// Resets everything back to the empty/welcome state
function clearThread() {
  threadId = null;
  isInterrupted = false;
  messagesEl.innerHTML = '';
  messagesEl.appendChild(emptyState.cloneNode(true));
  threadDisp.textContent = 'no thread';
  inputEl.value = '';
  autoResize(inputEl);
}

// ── Send message ──────────────────────────────────────────────────────────
// Posts the user's message to the backend and renders the response
async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || isLoading) return;

  hideEmpty();
  appendUserBubble(text);
  inputEl.value = '';
  autoResize(inputEl);
  setLoading(true);

  const typingEl = appendTyping();

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, thread_id: threadId || undefined }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    threadId = data.thread_id;
    threadDisp.textContent = threadId.slice(0, 18) + '…';
    isInterrupted = data.interrupted;

    typingEl.remove();

    if (data.interrupted && data.interrupt_data) {
      appendHITLCard(data.interrupt_data);
    } else {
      appendAIBubble(data.response, data.intent);
    }

  } catch (err) {
    typingEl.remove();
    appendError(err.message);
  } finally {
    setLoading(false);
  }
}

// ── HITL confirm / cancel ─────────────────────────────────────────────────
// Called when the user clicks Send or Cancel on the WhatsApp confirmation card
async function resolveHITL(confirmed) {
  if (!threadId || isLoading) return;

  document.querySelectorAll('.hitl-card').forEach(el => el.remove());

  appendUserBubble(confirmed ? '✓ Confirmed' : '✗ Cancelled');
  setLoading(true);
  const typingEl = appendTyping();

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: JSON.stringify({ confirmed }),
        thread_id: threadId,
      }),
    });

    const data = await res.json();
    isInterrupted = data.interrupted;
    typingEl.remove();

    if (data.interrupted && data.interrupt_data) {
      appendHITLCard(data.interrupt_data);
    } else {
      appendAIBubble(data.response, data.intent);
    }

  } catch (err) {
    typingEl.remove();
    appendError(err.message);
  } finally {
    setLoading(false);
  }
}

// ── DOM builders ──────────────────────────────────────────────────────────

function hideEmpty() {
  const e = document.getElementById('emptyState');
  if (e) e.remove();
}

function appendUserBubble(text) {
  const group = document.createElement('div');
  group.className = 'msg-group user';
  group.innerHTML = `
    <div class="msg-label">you</div>
    <div class="bubble user">${escHtml(text)}</div>
  `;
  messagesEl.appendChild(group);
  scrollBottom();
}

function appendAIBubble(text, intent) {
  const group = document.createElement('div');
  group.className = 'msg-group ai';

  const intentBadge = intent && intent !== 'unknown'
    ? `<div class="intent-badge ${intentClass(intent)}">${intent.replace('_', ' ')}</div>`
    : '';

  group.innerHTML = `
    <div class="msg-label">lifeops</div>
    <div class="bubble ai">${renderMarkdownLite(text)}</div>
    ${intentBadge}
  `;
  messagesEl.appendChild(group);
  scrollBottom();
}

function appendHITLCard(interruptData) {
  const to    = interruptData.to    || '—';
  const draft = interruptData.draft || '';

  const card = document.createElement('div');
  card.className = 'msg-group ai';
  card.innerHTML = `
    <div class="msg-label">lifeops · confirmation needed</div>
    <div class="hitl-card">
      <div class="hitl-header">
        <div class="hitl-icon">📱</div>
        <div>
          <div class="hitl-title">Send WhatsApp Message?</div>
          <div class="hitl-to">to: ${escHtml(to)}</div>
        </div>
      </div>
      <div class="hitl-draft">${escHtml(draft)}</div>
      <div class="hitl-actions">
        <button class="hitl-btn confirm" onclick="resolveHITL(true)">Send</button>
        <button class="hitl-btn cancel"  onclick="resolveHITL(false)">Cancel</button>
      </div>
    </div>
  `;
  messagesEl.appendChild(card);
  scrollBottom();
}

function appendTyping() {
  const wrap = document.createElement('div');
  wrap.className = 'msg-group ai';
  wrap.innerHTML = `
    <div class="msg-label">lifeops</div>
    <div class="typing">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>
  `;
  messagesEl.appendChild(wrap);
  scrollBottom();
  return wrap;
}

function appendError(msg) {
  const group = document.createElement('div');
  group.className = 'msg-group ai';
  group.innerHTML = `
    <div class="msg-label">error</div>
    <div class="bubble ai" style="color: var(--red); border-color: #3a2020;">
      ⚠ ${escHtml(msg)}
    </div>
  `;
  messagesEl.appendChild(group);
  scrollBottom();
}

// ── Helpers ───────────────────────────────────────────────────────────────

// Disable/enable input while waiting for a response
function setLoading(v) {
  isLoading = v;
  sendBtn.disabled = v;
  inputEl.disabled = v;
}

function scrollBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// Prevent raw user text from being interpreted as HTML
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// Map intent string to a CSS class for the coloured badge
function intentClass(intent) {
  if (intent?.includes('travel'))   return 'travel';
  if (intent?.includes('reminder')) return 'reminder';
  if (intent?.includes('creative')) return 'creative';
  return 'unknown';
}

// Converts **bold**, *italic*, `code`, and newlines into HTML
function renderMarkdownLite(text) {
  return escHtml(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,     '<em>$1</em>')
    .replace(/_(.+?)_/g,       '<em>$1</em>')
    .replace(/`(.+?)`/g,       '<code>$1</code>')
    .replace(/\n/g,            '<br>');
}