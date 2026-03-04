const SERVER_URL = 'http://localhost:8765';
const REQUEST_TIMEOUT_MS = 15 * 60 * 1000; // 15 minutes

// DOM elements
const serverStatus = document.getElementById('server-status');
const statusDot = serverStatus.querySelector('.status-dot');
const statusText = serverStatus.querySelector('.status-text');
const notYoutube = document.getElementById('not-youtube');
const controls = document.getElementById('controls');
const videoUrl = document.getElementById('video-url');
const modelSelect = document.getElementById('model');
const processBtn = document.getElementById('process-btn');
const progress = document.getElementById('progress');
const progressText = document.getElementById('progress-text');
const cancelBtn = document.getElementById('cancel-btn');
const results = document.getElementById('results');
const error = document.getElementById('error');
const errorMessage = document.getElementById('error-message');

// Tab elements
const tabs = document.querySelectorAll('.tab');
const summaryContent = document.getElementById('summary-content');
const transcriptContent = document.getElementById('transcript-content');
const quizContent = document.getElementById('quiz-content');
const copyBtn = document.getElementById('copy-btn');
const newBtn = document.getElementById('new-btn');
const retryBtn = document.getElementById('retry-btn');

let currentUrl = null;
let currentResults = null;
let currentTab = 'summary';
let abortController = null;

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
  await checkServerHealth();
  await checkCurrentTab();
  setupEventListeners();
});

async function checkServerHealth() {
  try {
    const response = await fetch(`${SERVER_URL}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(3000),
    });
    const data = await response.json();

    if (data.status === 'ok') {
      statusDot.classList.add('connected');
      statusDot.classList.remove('disconnected');
      if (data.openai_configured) {
        statusText.textContent = 'Server connected';
      } else {
        statusText.textContent = 'Server connected (API key missing)';
        statusDot.classList.remove('connected');
        statusDot.classList.add('disconnected');
      }
      return true;
    }
  } catch (e) {
    statusDot.classList.add('disconnected');
    statusDot.classList.remove('connected');
    statusText.textContent = 'Server not running';
    return false;
  }
}

async function checkCurrentTab() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (tab && tab.url && isYouTubeVideo(tab.url)) {
      currentUrl = tab.url;
      videoUrl.textContent = truncateUrl(tab.url, 50);
      controls.classList.remove('hidden');
      notYoutube.classList.add('hidden');
    } else {
      controls.classList.add('hidden');
      notYoutube.classList.remove('hidden');
    }
  } catch (e) {
    console.error('Error checking tab:', e);
    controls.classList.add('hidden');
    notYoutube.classList.remove('hidden');
  }
}

function isYouTubeVideo(url) {
  const patterns = [
    /youtube\.com\/watch\?v=/,
    /youtu\.be\//,
    /youtube\.com\/embed\//,
    /youtube\.com\/shorts\//,
  ];
  return patterns.some(p => p.test(url));
}

function truncateUrl(url, maxLength) {
  if (url.length <= maxLength) return url;
  return url.substring(0, maxLength) + '...';
}

function setupEventListeners() {
  processBtn.addEventListener('click', processVideo);
  retryBtn.addEventListener('click', processVideo);
  cancelBtn.addEventListener('click', cancelProcessing);
  newBtn.addEventListener('click', resetToControls);
  copyBtn.addEventListener('click', copyCurrentTab);

  tabs.forEach(tab => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
  });
}

async function processVideo() {
  if (!currentUrl) return;

  // Set up abort controller for timeout and cancellation
  abortController = new AbortController();
  const timeoutId = setTimeout(() => abortController.abort(), REQUEST_TIMEOUT_MS);

  // Show progress
  controls.classList.add('hidden');
  error.classList.add('hidden');
  results.classList.add('hidden');
  progress.classList.remove('hidden');
  progressText.textContent = 'Processing video...';

  try {
    const response = await fetch(`${SERVER_URL}/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url: currentUrl,
        model: modelSelect.value,
      }),
      signal: abortController.signal,
    });

    clearTimeout(timeoutId);
    const data = await response.json();

    if (data.success) {
      currentResults = data;
      showResults(data);
    } else {
      showError(data.error || 'Unknown error occurred');
    }
  } catch (e) {
    clearTimeout(timeoutId);
    if (e.name === 'AbortError') {
      showError('Request was cancelled.');
    } else {
      console.error('Error processing video:', e);
      if (e instanceof TypeError && e.message === 'Failed to fetch') {
        showError('Failed to connect to server. Is it running?');
      } else {
        showError(`Error: ${e.message}`);
      }
    }
  } finally {
    abortController = null;
  }
}

function cancelProcessing() {
  if (abortController) {
    abortController.abort();
  }
}

function renderMarkdown(text) {
  const lines = text.split('\n');
  const htmlParts = [];
  let listType = null; // 'ul', 'ol', or null
  let inCodeBlock = false;
  let codeLines = [];

  function closeList() {
    if (listType) { htmlParts.push(`</${listType}>`); listType = null; }
  }

  for (const line of lines) {
    const trimmed = line.trim();

    // Code blocks (``` ... ```)
    if (trimmed.startsWith('```')) {
      if (inCodeBlock) {
        htmlParts.push(`<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
        codeLines = [];
        inCodeBlock = false;
      } else {
        closeList();
        inCodeBlock = true;
      }
      continue;
    }
    if (inCodeBlock) {
      codeLines.push(line);
      continue;
    }

    // Horizontal rule
    if (/^---+$/.test(trimmed)) {
      closeList();
      htmlParts.push('<hr>');
      continue;
    }

    // Headers (### before ## before #)
    const headerMatch = trimmed.match(/^(#{1,6})\s+(.+)$/);
    if (headerMatch) {
      closeList();
      const level = headerMatch[1].length;
      htmlParts.push(`<h${level}>${applyInlineFormatting(headerMatch[2])}</h${level}>`);
      continue;
    }

    // Blockquotes
    if (trimmed.startsWith('> ')) {
      closeList();
      htmlParts.push(`<blockquote>${applyInlineFormatting(trimmed.slice(2))}</blockquote>`);
      continue;
    }

    // Unordered list items
    if (trimmed.startsWith('- ')) {
      if (listType !== 'ul') { closeList(); htmlParts.push('<ul>'); listType = 'ul'; }
      htmlParts.push(`<li>${applyInlineFormatting(trimmed.slice(2))}</li>`);
      continue;
    }

    // Ordered list items
    const olMatch = trimmed.match(/^\d+[.)]\s+(.+)$/);
    if (olMatch) {
      if (listType !== 'ol') { closeList(); htmlParts.push('<ol>'); listType = 'ol'; }
      htmlParts.push(`<li>${applyInlineFormatting(olMatch[1])}</li>`);
      continue;
    }

    // Close list if we hit a non-list line
    closeList();

    // Empty line
    if (trimmed === '') {
      continue;
    }

    // Paragraph
    htmlParts.push(`<p>${applyInlineFormatting(trimmed)}</p>`);
  }

  // Close any unclosed blocks
  if (inCodeBlock) {
    htmlParts.push(`<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
  }
  closeList();
  return htmlParts.join('');
}

function applyInlineFormatting(text) {
  let safe = escapeHtml(text);
  // Inline code: `text` (before other formatting so contents aren't processed)
  safe = safe.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Bold: **text**
  safe = safe.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic: *text*
  safe = safe.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  // Links: [text](url) — only allow http(s) to prevent javascript: XSS
  safe = safe.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, text, url) => {
    if (/^https?:\/\//i.test(url)) {
      return `<a href="${url}" target="_blank">${text}</a>`;
    }
    return match;
  });
  return safe;
}

const _escapeDiv = document.createElement('div');
function escapeHtml(text) {
  _escapeDiv.textContent = text;
  return _escapeDiv.innerHTML;
}

function showResults(data) {
  progress.classList.add('hidden');
  results.classList.remove('hidden');

  summaryContent.innerHTML = renderMarkdown(data.summary);
  transcriptContent.innerHTML = renderMarkdown(data.transcript);
  quizContent.innerHTML = renderMarkdown(data.quiz);

  // Reset to summary tab
  switchTab('summary');
}

function showError(message) {
  progress.classList.add('hidden');
  error.classList.remove('hidden');
  errorMessage.textContent = message;
}

function resetToControls() {
  results.classList.add('hidden');
  error.classList.add('hidden');
  progress.classList.add('hidden');
  controls.classList.remove('hidden');
  currentResults = null;
}

function switchTab(tabName) {
  currentTab = tabName;

  // Update tab buttons
  tabs.forEach(tab => {
    tab.classList.toggle('active', tab.dataset.tab === tabName);
  });

  // Update tab panes
  document.querySelectorAll('.tab-pane').forEach(pane => {
    pane.classList.add('hidden');
    pane.classList.remove('active');
  });

  const activePane = document.getElementById(`${tabName}-tab`);
  activePane.classList.remove('hidden');
  activePane.classList.add('active');
}

async function copyCurrentTab() {
  if (!currentResults) return;

  let content;
  switch (currentTab) {
    case 'summary':
      content = currentResults.summary;
      break;
    case 'transcript':
      content = currentResults.transcript;
      break;
    case 'quiz':
      content = currentResults.quiz;
      break;
  }

  try {
    await navigator.clipboard.writeText(content);
    const originalText = copyBtn.textContent;
    copyBtn.textContent = 'Copied!';
    setTimeout(() => {
      copyBtn.textContent = originalText;
    }, 1500);
  } catch (e) {
    console.error('Failed to copy:', e);
  }
}
