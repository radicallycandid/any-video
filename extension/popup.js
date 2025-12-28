const SERVER_URL = 'http://localhost:8765';

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
  newBtn.addEventListener('click', resetToControls);
  copyBtn.addEventListener('click', copyCurrentTab);

  tabs.forEach(tab => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
  });
}

async function processVideo() {
  if (!currentUrl) return;

  // Show progress
  controls.classList.add('hidden');
  error.classList.add('hidden');
  results.classList.add('hidden');
  progress.classList.remove('hidden');
  progressText.textContent = 'Downloading audio...';

  try {
    const response = await fetch(`${SERVER_URL}/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url: currentUrl,
        model: modelSelect.value,
      }),
    });

    const data = await response.json();

    if (data.success) {
      currentResults = data;
      showResults(data);
    } else {
      showError(data.error || 'Unknown error occurred');
    }
  } catch (e) {
    console.error('Error processing video:', e);
    showError(`Failed to connect to server: ${e.message}`);
  }
}

function showResults(data) {
  progress.classList.add('hidden');
  results.classList.remove('hidden');

  summaryContent.textContent = data.summary;
  transcriptContent.textContent = data.transcript;
  quizContent.textContent = data.quiz;

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
