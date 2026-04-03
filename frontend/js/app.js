// ── Config ────────────────────────────────────────────────────────────────
// In production, set window.__API_BASE__ before loading this script,
// or the API is on the same origin.
const API_BASE = window.__API_BASE__ || window.location.origin;

// ── State ─────────────────────────────────────────────────────────────────
let threadId = null;
let isLoading = false;
let isInterrupted = false;
let authToken = null;
let user = null;
let isRegisterMode = false;
let streamGeneration = 0; // incremented on each clearThread to cancel stale streams

// ── DOM refs ──────────────────────────────────────────────────────────────
const authScreen   = document.getElementById('authScreen');
const appEl        = document.getElementById('app');
const messagesEl   = document.getElementById('messages');
const inputEl      = document.getElementById('input');
const sendBtn      = document.getElementById('sendBtn');
const statusDot    = document.getElementById('statusDot');
const statusLabel  = document.getElementById('statusLabel');
const emptyState   = document.getElementById('emptyState');
const authError    = document.getElementById('authError');
const settingsModal = document.getElementById('settingsModal');
const toastContainer = document.getElementById('toastContainer');

// ── Toast System ──────────────────────────────────────────────────────────
function showToast({ type = 'info', title, message, actions = [], duration = 6000 }) {
  const toast = document.createElement('div');
  toast.className = 'toast';

  const icons = {
    info: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/></svg>',
    success: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>',
    warning: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/></svg>',
    error: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.47 2 2 6.47 2 12s4.47 10 10 10 10-4.47 10-10S17.53 2 12 2zm5 13.59L15.59 17 12 13.41 8.41 17 7 15.59 10.59 12 7 8.41 8.41 7 12 10.59 15.59 7 17 8.41 13.41 12 17 15.59z"/></svg>',
  };

  let actionsHtml = '';
  if (actions.length > 0) {
    actionsHtml = '<div class="toast-actions">' +
      actions.map(a => `<button class="toast-action-btn ${a.primary ? 'primary' : 'ghost'}" onclick="${a.onClick}">${a.label}</button>`).join('') +
      '</div>';
  }

  toast.innerHTML = `
    <div class="toast-icon ${type}">${icons[type]}</div>
    <div class="toast-body">
      <div class="toast-title">${title}</div>
      <div class="toast-message">${message}</div>
      ${actionsHtml}
    </div>
    <button class="toast-close" onclick="dismissToast(this.parentElement)">
      <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>
    </button>
  `;

  toastContainer.appendChild(toast);

  if (duration > 0) {
    setTimeout(() => dismissToast(toast), duration);
  }

  return toast;
}

function dismissToast(toast) {
  if (!toast || !toast.parentElement) return;
  toast.classList.add('toast-exit');
  setTimeout(() => toast.remove(), 200);
}

window.closeNearestToast = function(btn) {
  const toast = btn.closest('.toast');
  if (toast) dismissToast(toast);
};

// ── Init ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  const saved = localStorage.getItem('lifeops_token');
  const savedUser = localStorage.getItem('lifeops_user');
  if (saved && savedUser) {
    // Validate the token is still valid against the server
    try {
      const res = await fetch(`${API_BASE}/auth/me`, {
        headers: { 'Authorization': `Bearer ${saved}` },
      });
      if (res.ok) {
        authToken = saved;
        user = await res.json();
        localStorage.setItem('lifeops_user', JSON.stringify(user));
        showApp();
        return;
      }
    } catch (e) { /* server unreachable, clear stale data */ }
    // Token is stale — clear it
    localStorage.removeItem('lifeops_token');
    localStorage.removeItem('lifeops_user');
  }
  showAuth();
});

// ── Auth UI ───────────────────────────────────────────────────────────────
function showAuth() {
  authScreen.style.display = 'flex';
  appEl.style.display = 'none';
}

function showApp() {
  authScreen.style.display = 'none';
  appEl.style.display = 'flex';
  checkHealth();
  setInterval(checkHealth, 15000);

  // Show email reminder toast if not configured
  if (authToken) {
    setTimeout(() => checkEmailAndNotify(), 2000);
  }
}

async function checkEmailAndNotify() {
  try {
    const res = await fetch(`${API_BASE}/settings/email`, {
      headers: authHeaders(),
    });
    if (!res.ok) return;
    const data = await res.json();
    if (!data.configured) {
      showToast({
        type: 'info',
        title: 'Email Reminders',
        message: 'Set up your email to receive reminders for your travel plans.',
        actions: [
          { label: 'Set Up', primary: true, onClick: 'openSettingsFromToast()' },
          { label: 'Later', primary: false, onClick: 'closeNearestToast(this)' },
        ],
        duration: 0,
      });
    }
  } catch (e) { /* silent */ }
}

window.openSettingsFromToast = function() {
  document.querySelectorAll('.toast').forEach(t => dismissToast(t));
  toggleSettings();
};

function toggleAuthMode(e) {
  e.preventDefault();
  isRegisterMode = !isRegisterMode;
  document.getElementById('authTitle').textContent = isRegisterMode ? 'Create account' : 'Sign in';
  document.getElementById('authSubtitle').textContent = isRegisterMode
    ? 'Create an account to persist your conversations.'
    : 'Welcome back — your conversation will be restored.';
  document.getElementById('authBtn').textContent = isRegisterMode ? 'Create account' : 'Sign in';
  document.getElementById('authToggleText').textContent = isRegisterMode
    ? 'Already have an account?'
    : "Don't have an account?";
  document.getElementById('authToggleLink').textContent = isRegisterMode ? 'Sign in' : 'Create one';
  authError.textContent = '';
}

async function handleAuth(e) {
  e.preventDefault();
  const email = document.getElementById('authEmail').value.trim();
  const password = document.getElementById('authPassword').value;
  authError.textContent = '';

  const endpoint = isRegisterMode ? '/auth/register' : '/auth/login';
  try {
    const res = await fetch(`${API_BASE}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Auth failed');

    authToken = data.access_token;
    user = data.user;
    localStorage.setItem('lifeops_token', authToken);
    localStorage.setItem('lifeops_user', JSON.stringify(user));
    showApp();
  } catch (err) {
    authError.textContent = err.message;
  }
}

function skipAuth(e) {
  e.preventDefault();
  authToken = null;
  user = null;
  showApp();
}

function logout() {
  authToken = null;
  user = null;
  threadId = null;
  localStorage.removeItem('lifeops_token');
  localStorage.removeItem('lifeops_user');
  // Clear conversation UI
  messagesEl.innerHTML = '';
  const emptyClone = document.createElement('div');
  emptyClone.className = 'empty-state';
  emptyClone.id = 'emptyState';
  emptyClone.innerHTML = emptyState.innerHTML;
  messagesEl.appendChild(emptyClone);
  showAuth();
}

// ── Auth headers helper ───────────────────────────────────────────────────
function authHeaders() {
  const h = { 'Content-Type': 'application/json' };
  if (authToken) h['Authorization'] = `Bearer ${authToken}`;
  return h;
}

// ── Health check ──────────────────────────────────────────────────────────
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

// ── Textarea auto-resize ──────────────────────────────────────────────────
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 140) + 'px';
}

// ── Key handler ───────────────────────────────────────────────────────────
function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

// ── Suggestion fill ───────────────────────────────────────────────────────
function fillSuggestion(el) {
  inputEl.value = el.querySelector('.suggestion-text').textContent;
  autoResize(inputEl);
  inputEl.focus();
}

// ── Clear thread ──────────────────────────────────────────────────────────
async function clearThread() {
  streamGeneration++; // cancel any in-flight streaming

  if (threadId) {
    try {
      await fetch(`${API_BASE}/chat/${threadId}`, {
        method: 'DELETE',
        headers: authHeaders(),
      });
    } catch (e) {
      console.warn('Failed to delete thread:', e);
    }
  }

  threadId = null;
  isInterrupted = false;
  messagesEl.innerHTML = '';
  const emptyClone = document.createElement('div');
  emptyClone.className = 'empty-state';
  emptyClone.id = 'emptyState';
  emptyClone.innerHTML = emptyState.innerHTML;
  messagesEl.appendChild(emptyClone);
  inputEl.value = '';
  autoResize(inputEl);
}

// ── Send message (streaming) ──────────────────────────────────────────────
async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || isLoading) return;

  hideEmpty();
  appendUserBubble(text);
  inputEl.value = '';
  autoResize(inputEl);
  setLoading(true);

  const myGen = streamGeneration; // capture generation
  const streamBubble = createStreamBubble();
  let fullContent = '';

  try {
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
    const res = await fetch(`${API_BASE}/chat/stream`, {
      method: 'POST',
      headers: { ...authHeaders(), 'Accept': 'text/event-stream' },
      body: JSON.stringify({ message: text, thread_id: threadId || undefined, timezone: tz }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      // If clearThread was called during streaming, abort
      if (myGen !== streamGeneration) {
        reader.cancel();
        break;
      }

      buffer += decoder.decode(value, { stream: true });

      // Process complete SSE events (data: ...\n\n)
      const lines = buffer.split('\n');
      buffer = lines.pop(); // Keep incomplete line in buffer

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;

        // Check generation before processing each event
        if (myGen !== streamGeneration) break;

        const data = JSON.parse(line.slice(6));

        if (data.type === 'token') {
          fullContent += data.content;
          updateStreamBubble(streamBubble, fullContent);
        } else if (data.type === 'done') {
          threadId = data.thread_id;
          isInterrupted = data.interrupted;
          streamBubble.remove();

          if (data.interrupted && data.interrupt_data) {
            appendHITLCard(data.interrupt_data);
          } else if (data.response) {
            appendAIBubble(data.response, data.intent);
          }
        } else if (data.type === 'error') {
          streamBubble.remove();
          appendError(data.message);
        }
      }
    }

  } catch (err) {
    streamBubble.remove();
    appendError(err.message);
  } finally {
    setLoading(false);
  }
}

function createStreamBubble() {
  const group = document.createElement('div');
  group.className = 'msg-group ai';
  group.innerHTML = `
    <div class="msg-label">assistant</div>
    <div class="bubble ai stream-bubble"></div>
  `;
  messagesEl.appendChild(group);
  scrollBottom();
  return group;
}

function updateStreamBubble(group, content) {
  const bubble = group.querySelector('.stream-bubble');
  if (bubble) {
    bubble.innerHTML = renderMarkdown(content);
    scrollBottom();
  }
}

// ── HITL confirm / cancel (streaming) ─────────────────────────────────────
async function resolveHITL(confirmed) {
  if (!threadId || isLoading) return;

  document.querySelectorAll('.hitl-card').forEach(el => el.remove());

  appendUserBubble(confirmed ? '\u2713 Confirmed' : '\u2717 Cancelled');
  setLoading(true);

  const streamBubble = createStreamBubble();
  let fullContent = '';

  try {
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
    const res = await fetch(`${API_BASE}/chat/stream`, {
      method: 'POST',
      headers: { ...authHeaders(), 'Accept': 'text/event-stream' },
      body: JSON.stringify({
        message: JSON.stringify({ confirmed }),
        thread_id: threadId,
        timezone: tz
      }),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = JSON.parse(line.slice(6));

        if (data.type === 'token') {
          fullContent += data.content;
          updateStreamBubble(streamBubble, fullContent);
        } else if (data.type === 'done') {
          isInterrupted = data.interrupted;
          streamBubble.remove();
          if (data.interrupted && data.interrupt_data) {
            appendHITLCard(data.interrupt_data);
          } else if (data.response) {
            appendAIBubble(data.response, data.intent);
          }
        } else if (data.type === 'error') {
          streamBubble.remove();
          appendError(data.message);
        }
      }
    }

  } catch (err) {
    streamBubble.remove();
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
    <div class="msg-label">assistant</div>
    <div class="bubble ai">${renderMarkdown(text)}</div>
    ${intentBadge}
  `;
  messagesEl.appendChild(group);
  scrollBottom();
}

function appendHITLCard(interruptData) {
  const to    = interruptData.to    || '\u2014';
  const draft = interruptData.draft || '';

  const card = document.createElement('div');
  card.className = 'msg-group ai';
  card.innerHTML = `
    <div class="msg-label">confirmation needed</div>
    <div class="hitl-card">
      <div class="hitl-header">
        <div class="hitl-icon">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M20 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 4l-8 5-8-5V6l8 5 8-5v2z"/></svg>
        </div>
        <div>
          <div class="hitl-title">Send Email Reminder?</div>
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
    <div class="msg-label">assistant</div>
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
      ${escHtml(msg)}
    </div>
  `;
  messagesEl.appendChild(group);
  scrollBottom();
}

// ── Settings Modal ────────────────────────────────────────────────────────
function toggleSettings() {
  const modal = settingsModal;
  if (modal.style.display === 'none' || !modal.style.display) {
    modal.style.display = 'flex';
    loadEmailConfig();
    if (user) {
      document.getElementById('settingsUserEmail').textContent = `Signed in as ${user.email}`;
      document.getElementById('settingsAccountEmail').textContent = user.email;
    } else {
      document.getElementById('settingsUserEmail').textContent = 'Not signed in — settings require an account.';
      document.getElementById('settingsAccountEmail').textContent = '—';
    }
  } else {
    modal.style.display = 'none';
  }
}

function closeSettingsOutside(e) {
  if (e.target === settingsModal) toggleSettings();
}

async function loadEmailConfig() {
  if (!authToken) return;
  try {
    const res = await fetch(`${API_BASE}/settings/email`, {
      headers: authHeaders(),
    });
    if (!res.ok) return;
    const data = await res.json();

    const notConfigured = document.getElementById('emailNotConfigured');
    const configured = document.getElementById('emailConfigured');
    const emailDisplay = document.getElementById('emailDisplay');

    if (data.configured) {
      notConfigured.style.display = 'none';
      configured.style.display = 'block';
      emailDisplay.textContent = `Sending from: ${data.email}`;
    } else {
      notConfigured.style.display = 'block';
      configured.style.display = 'none';
    }
  } catch (e) {
    console.warn('Failed to load email config:', e);
  }
}

async function saveEmailConfig() {
  const appPassword = document.getElementById('smtpPassword').value.trim();
  const statusEl = document.getElementById('emailStatus');

  if (!appPassword) {
    statusEl.textContent = 'App password is required.';
    statusEl.className = 'settings-status error';
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/settings/email`, {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify({ app_password: appPassword }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Failed to save');

    statusEl.textContent = 'Email settings saved!';
    statusEl.className = 'settings-status success';
    document.getElementById('smtpPassword').value = '';
    loadEmailConfig();
  } catch (err) {
    statusEl.textContent = err.message;
    statusEl.className = 'settings-status error';
  }
}

async function removeEmailConfig() {
  try {
    await fetch(`${API_BASE}/settings/email`, {
      method: 'DELETE',
      headers: authHeaders(),
    });
    document.getElementById('emailNotConfigured').style.display = 'block';
    document.getElementById('emailConfigured').style.display = 'none';
    document.getElementById('emailStatus').textContent = 'Email config removed.';
    document.getElementById('emailStatus').className = 'settings-status';
  } catch (err) {
    showToast({ type: 'error', title: 'Error', message: err.message });
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────

function setLoading(v) {
  isLoading = v;
  sendBtn.disabled = v;
  inputEl.disabled = v;
}

function scrollBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function intentClass(intent) {
  if (intent?.includes('travel'))   return 'travel';
  if (intent?.includes('reminder')) return 'reminder';
  if (intent?.includes('creative')) return 'creative';
  return 'unknown';
}

// ── Markdown renderer ─────────────────────────────────────────────────────
function renderMarkdown(text) {
  let html = escHtml(text);

  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  html = html.replace(/_(.+?)_/g, '<em>$1</em>');
  html = html.replace(/^[\-\*] (.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>\n?)+/g, (match) => `<ul>${match}</ul>`);
  html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
  html = html.replace(/\n(?!<li>|<\/li>|<ul>|<\/ul>)/g, '<br>');

  return html;
}
