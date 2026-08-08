// AI Technical Interview Agent — Client App Logic (Accessibility-first revision)
'use strict';

let state = {
  candidates: [],
  selectedCandidate: null,
  sessionId: null,
  questionsAsked: 0,
  daysCovered: [],
  isSpeaking: false,
  isListening: false,
  ttsEnabled: true,
  speechRate: 1.0,
  fontSize: 'normal',
  contrastMode: 'normal',
  byokApiKey: '',
  selectedModel: 'gemma-4-31b-it',
  // Track last focused element before opening a modal (for focus-restore on close)
  _lastFocusBeforeModal: null,
  // Track which modal is open so hotkeys can guard against modal-open state
  _openModalId: null,
};

// ─── DOM References ────────────────────────────────────────────────────────────
const views = {
  candidateSelect: document.getElementById('view-candidate-select'),
  interviewRoom: document.getElementById('view-interview-room'),
  feedbackPortal: document.getElementById('view-feedback-portal'),
};

// ─── Screen-reader announcement helper ────────────────────────────────────────
function announceSR(message) {
  const container = document.getElementById('sr-announcements');
  if (container) {
    container.textContent = '';
    setTimeout(() => { container.textContent = message; }, 50);
  }
}

// ─── Toast notification (replaces alert() — fixes C6, C7) ─────────────────────
let _toastTimeout = null;
function showToast(message, type = 'info') {
  let toast = document.getElementById('a11y-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'a11y-toast';
    toast.setAttribute('role', 'status');
    toast.setAttribute('aria-live', 'polite');
    toast.setAttribute('aria-atomic', 'true');
    document.body.appendChild(toast);
  }
  toast.className = `a11y-toast a11y-toast--${type}`;
  toast.textContent = message;
  toast.classList.remove('a11y-toast--hidden');
  clearTimeout(_toastTimeout);
  _toastTimeout = setTimeout(() => toast.classList.add('a11y-toast--hidden'), 4000);
  announceSR(message);
}

// ─── View switching ────────────────────────────────────────────────────────────
function showView(viewName) {
  Object.keys(views).forEach(v => {
    const isTarget = v === viewName;
    views[v].classList.toggle('active', isTarget);
    if (isTarget) {
      // Move focus to the section heading for screen-reader context
      const heading = views[v].querySelector('h2');
      if (heading) {
        heading.setAttribute('tabindex', '-1');
        heading.focus();
      } else {
        views[v].focus();
      }
    }
  });
}

// ─── Focus-trap utility (fixes C1, C2, C3) ────────────────────────────────────
const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

function openModal(modalId, triggerEl) {
  const modal = document.getElementById(modalId);
  if (!modal) return;

  // Remember who triggered this modal so we can restore focus on close
  state._lastFocusBeforeModal = triggerEl || document.activeElement;
  state._openModalId = modalId;

  modal.classList.remove('hidden');

  // Move focus to the modal box itself
  const box = modal.querySelector('.modal-box');
  if (box) box.focus();

  // Trap focus inside modal
  modal._trapHandler = (e) => {
    if (e.key !== 'Tab') return;
    const focusableEls = Array.from(modal.querySelectorAll(FOCUSABLE));
    if (!focusableEls.length) return;
    const first = focusableEls[0];
    const last = focusableEls[focusableEls.length - 1];
    if (e.shiftKey) {
      if (document.activeElement === first) { e.preventDefault(); last.focus(); }
    } else {
      if (document.activeElement === last) { e.preventDefault(); first.focus(); }
    }
  };
  modal.addEventListener('keydown', modal._trapHandler);

  // Close on Esc
  modal._escHandler = (e) => {
    if (e.key === 'Escape') closeModal(modalId);
  };
  modal.addEventListener('keydown', modal._escHandler);

  // Close on backdrop click (but not on modal-box click)
  modal._backdropClickHandler = (e) => {
    if (e.target === modal) closeModal(modalId);
  };
  modal.addEventListener('click', modal._backdropClickHandler);
}

function closeModal(modalId) {
  const modal = document.getElementById(modalId);
  if (!modal) return;

  modal.classList.add('hidden');

  // Remove event handlers
  if (modal._trapHandler) modal.removeEventListener('keydown', modal._trapHandler);
  if (modal._escHandler) modal.removeEventListener('keydown', modal._escHandler);
  if (modal._backdropClickHandler) modal.removeEventListener('click', modal._backdropClickHandler);

  state._openModalId = null;

  // Return focus to the element that triggered the modal (fixes C3)
  if (state._lastFocusBeforeModal && typeof state._lastFocusBeforeModal.focus === 'function') {
    state._lastFocusBeforeModal.focus();
    state._lastFocusBeforeModal = null;
  }
}

// ─── Fetch candidates ─────────────────────────────────────────────────────────
async function fetchCandidates() {
  const container = document.getElementById('candidates-grid');
  try {
    const res = await fetch('/api/candidates');
    const candidates = await res.json();
    state.candidates = candidates;
    renderCandidates(candidates);
  } catch (err) {
    console.error('Failed to load candidates:', err);
    if (container) {
      container.innerHTML = `<p class="error-msg" role="alert">Failed to load cohort candidate profiles. Check backend connection.</p>`;
    }
  }
}

// ─── Render candidate cards ───────────────────────────────────────────────────
function renderCandidates(candidatesList) {
  const container = document.getElementById('candidates-grid');
  if (!container) return;

  if (candidatesList.length === 0) {
    container.innerHTML = `<p role="status">No matching candidate profiles found.</p>`;
    return;
  }

  // Build DOM nodes — avoid innerHTML with untrusted data (partial fix for M6)
  const fragment = document.createDocumentFragment();

  candidatesList.forEach(c => {
    const mem = c.member;
    const sig = c.signals || {};
    const firstTryPercent = Math.round((sig.missionsFirstTry / (sig.missionsCompleted || 1)) * 100);

    const article = document.createElement('article');
    article.className = 'candidate-card';
    // Fix S1: article is naturally a sectioning element; do not mix role="listitem" inside role="list"
    // The container role="list" is removed from HTML; articles flow as a grid naturally.
    article.setAttribute('tabindex', '0');
    article.setAttribute('aria-label', `${mem.name}, ${mem.jobRole}`);

    article.innerHTML = `
      <div class="candidate-card-header">
        <div>
          <h3 class="cand-name">${escapeHtml(mem.name)}</h3>
          <span class="cand-role">${escapeHtml(mem.jobRole)}</span>
        </div>
        <span class="badge-role">${escapeHtml(String(mem.yearsExperience))} Yrs Exp</span>
      </div>
      <ul class="cand-stats-list" aria-label="Learning Journey Metrics">
        <li><span>Missions Completed:</span> <strong>${escapeHtml(String(sig.missionsCompleted))} / 31 Days</strong></li>
        <li><span>First-Try Pass Rate:</span> <strong>${firstTryPercent}%</strong></li>
        <li><span>Commit Frequency:</span> <strong>${escapeHtml(String(sig.commitDays))} Days</strong></li>
        <li><span>Education:</span> <strong>${escapeHtml(mem.education)}</strong></li>
      </ul>
      <button class="btn-start-interview" data-id="${escapeHtml(mem.id)}" aria-label="Start Technical Interview for ${escapeHtml(mem.name)}">
        Start Technical Interview ➔
      </button>
    `;

    article.querySelector('.btn-start-interview').addEventListener('click', (e) => {
      startInterviewSession(e.currentTarget.getAttribute('data-id'));
    });

    // Allow Enter/Space on article to activate the card's button
    article.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        article.querySelector('.btn-start-interview').click();
      }
    });

    fragment.appendChild(article);
  });

  container.innerHTML = '';
  container.appendChild(fragment);
}

// ─── XSS-safe HTML escaping (fixes M6) ────────────────────────────────────────
function escapeHtml(str) {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(String(str ?? '')));
  return div.innerHTML;
}

// ─── Filter / search ──────────────────────────────────────────────────────────
function setupSearchAndFilters() {
  const searchInput = document.getElementById('candidate-search-input');
  const filterPills = document.querySelectorAll('.pill-filter');

  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      const query = e.target.value.toLowerCase();
      const filtered = state.candidates.filter(c =>
        c.member.name.toLowerCase().includes(query) ||
        c.member.jobRole.toLowerCase().includes(query) ||
        c.member.id.toLowerCase().includes(query)
      );
      renderCandidates(filtered);
      announceSR(`${filtered.length} candidates shown.`);
    });
  }

  filterPills.forEach(pill => {
    // Fix S3: set aria-pressed initially
    pill.setAttribute('aria-pressed', pill.classList.contains('active') ? 'true' : 'false');

    pill.addEventListener('click', (e) => {
      filterPills.forEach(p => {
        p.classList.remove('active');
        p.setAttribute('aria-pressed', 'false');
      });
      e.currentTarget.classList.add('active');
      e.currentTarget.setAttribute('aria-pressed', 'true');

      const filter = e.currentTarget.getAttribute('data-filter');
      let filtered = state.candidates;

      if (filter === 'senior') filtered = state.candidates.filter(c => c.member.yearsExperience >= 5);
      else if (filter === 'junior') filtered = state.candidates.filter(c => c.member.yearsExperience < 5);
      else if (filter === 'perfect') filtered = state.candidates.filter(c => (c.signals?.missionsCompleted || 0) >= 30);

      renderCandidates(filtered);
      announceSR(`Filtered to ${filtered.length} candidates.`);
    });
  });
}

function updateModeBadge(mode, notice) {
  const badge = document.getElementById('mode-status-badge');
  if (!badge) return;

  if (mode === 'live') {
    badge.className = 'badge-mode live';
    badge.textContent = '⚡ Live AI Mode (Google AI Studio)';
  } else if (mode === 'fallback') {
    badge.className = 'badge-mode fallback';
    badge.textContent = '⚠️ Live Mode — Fallback Active';
  } else {
    badge.className = 'badge-mode demo';
    badge.textContent = '💡 Demo Mode (Simulated AI)';
  }
  badge.title = notice || '';
}

// ─── Start interview session ───────────────────────────────────────────────────
async function startInterviewSession(candidateId) {
  try {
    const res = await fetch('/api/interview/start', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-GEMINI-API-KEY': state.byokApiKey,
      },
      body: JSON.stringify({ candidate_id: candidateId }),
    });

    const data = await res.json();
    state.sessionId = data.session_id;
    state.selectedCandidate = data.candidate;
    state.questionsAsked = data.questions_asked;
    state.daysCovered = data.days_covered_list;

    document.getElementById('heading-interview-room').textContent = `${data.candidate.name}'s Interview`;
    document.getElementById('candidate-role-badge').textContent = data.candidate.jobRole;
    updateModeBadge(data.mode, data.mode_notice);

    updateProgressGauges(data.questions_asked, data.days_covered_list, data.current_topic);

    const chatLog = document.getElementById('chat-messages-log');
    chatLog.innerHTML = '';
    appendMessage('agent', data.initial_question);

    showView('interviewRoom');
    announceSR(`Interview session started for ${data.candidate.name}. ${data.mode_notice || ''}`);

    if (state.ttsEnabled) speakText(data.initial_question);
  } catch (err) {
    console.error('Failed to start interview:', err);
    showToast('Error starting interview session. Please check backend connection.', 'error');
  }
}

// ─── Progress gauges ──────────────────────────────────────────────────────────
function updateProgressGauges(questionsCount, daysList, currentTopic) {
  document.getElementById('gauge-questions-count').textContent = `Q${questionsCount} / 8+`;
  document.getElementById('gauge-days-count').textContent = `${daysList.length} / 4+ Days`;

  if (currentTopic) {
    document.getElementById('current-topic-badge').textContent = currentTopic;
  }

  const container = document.getElementById('days-badges-container');
  if (container) {
    // Fix S5: use <span role="listitem"> inside a region labelled as a list
    container.innerHTML = daysList.map(d =>
      `<span class="day-chip" role="listitem">Day ${escapeHtml(String(d))}</span>`
    ).join('');
  }
}

// ─── Append chat message (XSS-safe, fix M6) ───────────────────────────────────
function appendMessage(role, text) {
  const chatLog = document.getElementById('chat-messages-log');
  if (!chatLog) return;

  const bubble = document.createElement('div');
  bubble.className = `chat-bubble ${role}`;
  bubble.setAttribute('tabindex', '0');

  const labelText = role === 'agent' ? 'AI Lead Architect' : 'Candidate Response';

  // Fix M1: visually-hidden role label distinguishes agent vs candidate for keyboard/SR users
  const meta = document.createElement('div');
  meta.className = 'bubble-meta';
  const emojiSpan = document.createElement('span');
  emojiSpan.setAttribute('aria-hidden', 'true');
  emojiSpan.textContent = role === 'agent' ? '🤖 ' : '👤 ';
  meta.appendChild(emojiSpan);
  meta.appendChild(document.createTextNode(labelText));

  const content = document.createElement('div');
  content.className = 'bubble-text';
  // Safe render: split on newlines, create text nodes + <br> elements
  const lines = text.split('\n');
  lines.forEach((line, i) => {
    content.appendChild(document.createTextNode(line));
    if (i < lines.length - 1) content.appendChild(document.createElement('br'));
  });

  bubble.appendChild(meta);
  bubble.appendChild(content);
  chatLog.appendChild(bubble);
  chatLog.scrollTop = chatLog.scrollHeight;
}

// ─── Submit answer ─────────────────────────────────────────────────────────────
async function submitAnswer() {
  const inputEl = document.getElementById('candidate-input-text');
  const text = inputEl.value.trim();
  if (!text || !state.sessionId) return;

  appendMessage('candidate', text);
  inputEl.value = '';

  try {
    const res = await fetch('/api/interview/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-GEMINI-API-KEY': state.byokApiKey,
      },
      body: JSON.stringify({ session_id: state.sessionId, message: text }),
    });

    const data = await res.json();
    state.questionsAsked = data.questions_asked;
    state.daysCovered = data.days_covered_list;

    updateProgressGauges(data.questions_asked, data.days_covered_list);

    // Update mode badge: if backend fell back to scripted response despite a live key, show warning
    if (data.fallback) {
      const reason = data.fallback_reason ? ` (${data.fallback_reason})` : '';
      showToast(`⚠️ Gemini API unavailable — using scripted fallback response${reason}`, 'warning');
      updateModeBadge('fallback', `Live mode active but Gemini API call failed${reason}. Response was generated using scripted fallback.`);
    } else if (data.mode) {
      updateModeBadge(data.mode, data.mode_notice);
    }

    appendMessage('agent', data.agent_response);

    if (data.live_test_challenge) openLiveTestModal(data.live_test_challenge);
    if (state.ttsEnabled) speakText(data.agent_response);
    if (data.is_complete) setTimeout(() => fetchFeedbackReport(), 1500);
  } catch (err) {
    console.error('Error submitting answer:', err);
    showToast('Failed to submit your answer. Please check your connection.', 'error');
  }
}

// ─── TTS ──────────────────────────────────────────────────────────────────────
function speakText(text) {
  if (!('speechSynthesis' in window)) return;

  window.speechSynthesis.cancel();
  const cleanText = text.replace(/[*#_`]/g, '');
  const utterance = new SpeechSynthesisUtterance(cleanText);
  utterance.rate = state.speechRate;

  const statusDot = document.getElementById('speaker-status-indicator');
  const statusText = document.getElementById('speaker-status-text');

  utterance.onstart = () => {
    state.isSpeaking = true;
    if (statusDot) statusDot.classList.add('speaking');
    if (statusText) statusText.textContent = 'AI Interviewer Speaking...';
  };
  utterance.onend = () => {
    state.isSpeaking = false;
    if (statusDot) statusDot.classList.remove('speaking');
    if (statusText) statusText.textContent = 'AI Interviewer Idle';
  };

  window.speechSynthesis.speak(utterance);
}

// ─── STT ──────────────────────────────────────────────────────────────────────
function setupSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    console.warn('Speech recognition not supported in this browser.');
    const micBtn = document.getElementById('btn-toggle-mic');
    if (micBtn) {
      micBtn.setAttribute('disabled', 'true');
      micBtn.setAttribute('aria-label', 'Voice input unavailable in this browser');
      micBtn.title = 'Voice input not supported';
    }
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = true;

  const micBtn = document.getElementById('btn-toggle-mic');
  const statusBar = document.getElementById('stt-status-bar');
  const inputText = document.getElementById('candidate-input-text');

  micBtn.addEventListener('click', () => {
    if (state.isListening) recognition.stop();
    else recognition.start();
  });

  recognition.onstart = () => {
    state.isListening = true;
    micBtn.classList.add('listening');
    micBtn.setAttribute('aria-pressed', 'true');
    micBtn.setAttribute('aria-label', 'Stop Voice Dictation (Alt + S)');
    statusBar.classList.remove('hidden');
    announceSR('Microphone active. Listening for candidate response.');
  };
  recognition.onresult = (event) => {
    inputText.value = Array.from(event.results).map(r => r[0].transcript).join('');
  };
  recognition.onend = () => {
    state.isListening = false;
    micBtn.classList.remove('listening');
    micBtn.setAttribute('aria-pressed', 'false');
    micBtn.setAttribute('aria-label', 'Start Voice Dictation (Alt + S)');
    statusBar.classList.add('hidden');
  };

  // Stop-button inside the STT status bar
  const stopBtn = document.getElementById('btn-stop-stt');
  if (stopBtn) stopBtn.addEventListener('click', () => recognition.stop());
}

// ─── Fetch feedback report ────────────────────────────────────────────────────
async function fetchFeedbackReport() {
  if (!state.sessionId) return;
  try {
    const res = await fetch(`/api/interview/session/${state.sessionId}/feedback`);
    const report = await res.json();
    renderFeedbackReport(report);
    showView('feedbackPortal');
    announceSR('Interview complete. Evaluation feedback report displayed.');
  } catch (err) {
    console.error('Failed to fetch feedback report:', err);
    showToast('Could not retrieve feedback report. Please try again.', 'error');
  }
}

// ─── Render feedback report ───────────────────────────────────────────────────
function renderFeedbackReport(report) {
  document.getElementById('report-candidate-subtitle').textContent =
    `Candidate: ${report.candidate.name} | ${report.candidate.jobRole}`;
  document.getElementById('report-overall-score').textContent = report.overall_score;
  document.getElementById('report-readiness-level').textContent = report.readiness_level;
  document.getElementById('report-recommendation-text').textContent = report.recommendation;
  document.getElementById('report-questions-total').textContent = `${report.questions_asked_total} Questions`;
  document.getElementById('report-days-total').textContent = `${report.unique_days_covered_count} Days`;

  // Score status tag
  const scoreTag = document.getElementById('report-score-status');
  if (scoreTag) {
    const passed = report.overall_score >= 70;
    scoreTag.textContent = passed ? 'PASSED' : 'NEEDS REVIEW';
    scoreTag.className = `score-tag ${passed ? 'pass' : ''}`;
  }

  // Domain table
  const tbody = document.getElementById('domain-table-body');
  if (tbody && report.domain_scores) {
    tbody.innerHTML = Object.entries(report.domain_scores).map(([domain, score]) => `
      <tr>
        <th scope="row">${escapeHtml(domain)}</th>
        <td><strong>${escapeHtml(String(score))} / 100</strong></td>
        <td><span class="score-tag ${score >= 80 ? 'pass' : ''}">${score >= 80 ? 'MASTERY' : 'REVIEW'}</span></td>
      </tr>
    `).join('');
  }

  const strengthsList = document.getElementById('report-strengths-list');
  if (strengthsList && report.strengths) {
    strengthsList.innerHTML = report.strengths.map(s => `<li>${escapeHtml(s)}</li>`).join('');
  }

  const growthList = document.getElementById('report-growth-list');
  if (growthList && report.growth_areas) {
    growthList.innerHTML = report.growth_areas.map(g => `<li>${escapeHtml(g)}</li>`).join('');
  }

  const reviewContainer = document.getElementById('report-review-days-container');
  if (reviewContainer && report.recommended_review_days) {
    reviewContainer.innerHTML = report.recommended_review_days.map(day => `
      <div class="card-box" style="padding: 1rem; margin-bottom: 0;">
        <span class="badge-role">Day ${escapeHtml(String(day))}</span>
        <h4 style="margin-top: 0.5rem;">Targeted Curriculum Day ${escapeHtml(String(day))}</h4>
        <p style="font-size: 0.85rem; color: var(--text-muted);">Revisit concepts and hands-on exercises for Day ${escapeHtml(String(day))}.</p>
      </div>
    `).join('');
  }
}

// ─── Live code testing modal ──────────────────────────────────────────────────
let liveTimerInterval = null;

function openLiveTestModal(challenge) {
  const modal = document.getElementById('modal-live-test');
  if (!modal || !challenge) return;

  document.getElementById('live-test-domain-badge').textContent = `Day ${challenge.day} Challenge`;
  document.getElementById('modal-live-test-title').textContent = `💻 Live Code: ${challenge.title}`;
  document.getElementById('live-test-problem').textContent = challenge.problem_statement;
  document.getElementById('live-test-code-editor').value = challenge.starter_code;

  const feedbackBox = document.getElementById('live-test-feedback-box');
  feedbackBox.classList.add('hidden');
  feedbackBox.textContent = '';

  // Timer (fix C5): update aria-label on the timer element so screen readers get it
  let timeRemaining = challenge.time_limit_seconds || 300;
  const timerEl = document.getElementById('live-test-timer');

  function formatTime(secs) {
    const m = String(Math.floor(secs / 60)).padStart(2, '0');
    const s = String(secs % 60).padStart(2, '0');
    return `${m}:${s}`;
  }

  clearInterval(liveTimerInterval);
  timerEl.textContent = `⏱️ ${formatTime(timeRemaining)}`;
  timerEl.setAttribute('aria-label', `Time remaining: ${timeRemaining} seconds`);

  liveTimerInterval = setInterval(() => {
    timeRemaining--;
    const formatted = formatTime(timeRemaining);
    timerEl.textContent = `⏱️ ${formatted}`;
    // Update aria-label every 30 s and at critical thresholds (60 s, 30 s, 0 s)
    if (timeRemaining % 30 === 0 || timeRemaining <= 60) {
      timerEl.setAttribute('aria-label', `Time remaining: ${timeRemaining} seconds`);
      if (timeRemaining <= 30 && timeRemaining > 0) {
        announceSR(`Warning: ${timeRemaining} seconds remaining on live code challenge.`);
      }
    }
    if (timeRemaining <= 0) {
      clearInterval(liveTimerInterval);
      timerEl.textContent = '⏱️ 00:00';
      timerEl.setAttribute('aria-label', 'Time is up');
      announceSR('Time is up for the live code challenge.');
    }
  }, 1000);

  // Open with focus-trap (fixes C1, C2)
  openModal('modal-live-test', document.getElementById('btn-open-settings')); // fallback trigger
  announceSR(`Live code challenge triggered for Day ${challenge.day}: ${challenge.title}. Timer started.`);
  document.getElementById('live-test-code-editor').focus();

  // Code submit handler
  document.getElementById('btn-submit-live-code').onclick = async () => {
    const code = document.getElementById('live-test-code-editor').value;
    try {
      const res = await fetch('/api/interview/live-test/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: state.sessionId, test_id: challenge.id, code }),
      });

      const result = await res.json();
      feedbackBox.classList.remove('hidden');

      // Fix C4: result communicated by text label + ARIA, not colour alone
      if (result.passed) {
        feedbackBox.className = 'live-test-feedback live-test-feedback--pass';
        feedbackBox.setAttribute('role', 'status');
        feedbackBox.textContent = `PASSED — Score: ${result.score}/100. ${result.feedback}`;
        announceSR(`Code challenge passed. Score: ${result.score} out of 100.`);
      } else {
        feedbackBox.className = 'live-test-feedback live-test-feedback--fail';
        feedbackBox.setAttribute('role', 'alert');
        feedbackBox.textContent = `NOT PASSED — ${result.feedback}`;
        announceSR(`Code challenge not passed. ${result.feedback}`);
      }

      setTimeout(() => {
        clearInterval(liveTimerInterval);
        closeModal('modal-live-test');
        appendMessage('candidate', `[Submitted Live Code Challenge: ${challenge.title}]`);
      }, 2500);
    } catch (err) {
      console.error('Error submitting code:', err);
      showToast('Code submission failed. Please try again.', 'error');
    }
  };
}

// ─── Accessibility controls setup ─────────────────────────────────────────────
function setupAccessibilityControls() {
  const fontInc = document.getElementById('btn-font-inc');
  const fontDec = document.getElementById('btn-font-dec');
  const fontIndicator = document.getElementById('font-size-indicator');
  const contrastBtn = document.getElementById('btn-toggle-contrast');
  const ttsBtn = document.getElementById('btn-toggle-tts');
  const rateSelect = document.getElementById('speech-rate-select');

  // ── Settings modal ───────────────────────────────────────────────────────
  const settingsBtn = document.getElementById('btn-open-settings');
  if (settingsBtn) {
    settingsBtn.addEventListener('click', () => openModal('modal-settings', settingsBtn));
  }
  document.getElementById('btn-close-settings')?.addEventListener('click', () => closeModal('modal-settings'));

  document.getElementById('btn-save-settings')?.addEventListener('click', async () => {
    const apiKey = document.getElementById('input-byok-key').value.trim();
    const model = document.getElementById('select-model-name').value;
    state.byokApiKey = apiKey;
    state.selectedModel = model;

    if (apiKey) {
      try {
        await fetch('/api/config/byok', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ api_key: apiKey, model_name: model }),
        });
      } catch (err) { console.error(err); }
    }

    closeModal('modal-settings');
    // Fix C7: replace alert() with toast
    showToast(`Settings saved! Active model: ${model}`, 'success');
  });

  // ── Live test modal close ────────────────────────────────────────────────
  document.getElementById('btn-close-live-test')?.addEventListener('click', () => {
    clearInterval(liveTimerInterval);
    closeModal('modal-live-test');
  });

  // ── Keyboard shortcuts modal ─────────────────────────────────────────────
  const helpBtn = document.getElementById('btn-keyboard-help');
  if (helpBtn) {
    helpBtn.addEventListener('click', () => openModal('modal-keyboard-help', helpBtn));
  }
  document.getElementById('btn-close-modal')?.addEventListener('click', () => closeModal('modal-keyboard-help'));

  // ── Font size controls ───────────────────────────────────────────────────
  function setFontSize(label, scale) {
    document.documentElement.setAttribute('data-size', scale);
    if (fontIndicator) {
      fontIndicator.textContent = label;
      // Fix M2: update aria-label dynamically
      fontIndicator.setAttribute('aria-label', `Current font scale: ${label}`);
    }
  }

  if (fontInc) {
    fontInc.addEventListener('click', () => setFontSize('125%', 'large'));
  }
  if (fontDec) {
    fontDec.addEventListener('click', () => setFontSize('100%', 'normal'));
  }

  // ── High contrast toggle ──────────────────────────────────────────────────
  if (contrastBtn) {
    // Fix S2: set initial aria-pressed
    contrastBtn.setAttribute('aria-pressed', 'false');
    contrastBtn.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-contrast');
      const next = current === 'high' ? 'normal' : 'high';
      document.documentElement.setAttribute('data-contrast', next);
      const isHigh = next === 'high';
      contrastBtn.classList.toggle('active', isHigh);
      contrastBtn.setAttribute('aria-pressed', String(isHigh));
      announceSR(`High contrast mode ${isHigh ? 'enabled' : 'disabled'}.`);
    });
  }

  // ── TTS toggle ────────────────────────────────────────────────────────────
  if (ttsBtn) {
    // Fix S2: set initial aria-pressed
    ttsBtn.setAttribute('aria-pressed', 'true');
    ttsBtn.addEventListener('click', () => {
      state.ttsEnabled = !state.ttsEnabled;
      ttsBtn.classList.toggle('active', state.ttsEnabled);
      ttsBtn.setAttribute('aria-pressed', String(state.ttsEnabled));
      ttsBtn.querySelector('.label').textContent = `Voice TTS: ${state.ttsEnabled ? 'ON' : 'OFF'}`;
      announceSR(`Auto read-aloud ${state.ttsEnabled ? 'enabled' : 'disabled'}.`);
    });
  }

  if (rateSelect) {
    rateSelect.addEventListener('change', (e) => {
      state.speechRate = parseFloat(e.target.value);
    });
  }

  // ── Global keyboard hotkeys ───────────────────────────────────────────────
  // Fix S8: guard against modal-open state
  document.addEventListener('keydown', (e) => {
    const modalOpen = state._openModalId !== null;

    if (e.altKey && e.key.toLowerCase() === 's') {
      if (modalOpen) return;
      e.preventDefault();
      document.getElementById('btn-toggle-mic')?.click();
    } else if (e.altKey && e.key.toLowerCase() === 'r') {
      if (modalOpen) return;
      e.preventDefault();
      const agentMsgs = document.querySelectorAll('.chat-bubble.agent');
      if (agentMsgs.length > 0) speakText(agentMsgs[agentMsgs.length - 1].textContent);
    } else if (e.altKey && e.key.toLowerCase() === 'c') {
      if (modalOpen) return;
      e.preventDefault();
      contrastBtn?.click();
    } else if (e.altKey && e.key === '+') {
      if (modalOpen) return;
      e.preventDefault();
      fontInc?.click();
    } else if (e.altKey && e.key === '-') {
      if (modalOpen) return;
      e.preventDefault();
      fontDec?.click();
    } else if (e.key === 'Enter' && (e.ctrlKey || document.activeElement?.id === 'candidate-input-text')) {
      if (modalOpen) return;
      if (!e.shiftKey) {
        e.preventDefault();
        submitAnswer();
      }
    } else if (e.altKey && e.key.toLowerCase() === 'f') {
      if (modalOpen) return;
      e.preventDefault();
      fetchFeedbackReport();
    }
    // Note: Esc is handled inside each modal's own escHandler (openModal)
  });
}

// ─── Button event listeners ───────────────────────────────────────────────────
function setupEventListeners() {
  document.getElementById('btn-submit-answer')?.addEventListener('click', submitAnswer);

  document.getElementById('btn-speak-question')?.addEventListener('click', () => {
    const agentMsgs = document.querySelectorAll('.chat-bubble.agent');
    if (agentMsgs.length > 0) speakText(agentMsgs[agentMsgs.length - 1].textContent);
  });

  document.getElementById('btn-back-to-list')?.addEventListener('click', () => showView('candidateSelect'));
  document.getElementById('btn-force-finish')?.addEventListener('click', fetchFeedbackReport);
  document.getElementById('btn-restart-new')?.addEventListener('click', () => showView('candidateSelect'));
  document.getElementById('btn-print-report')?.addEventListener('click', () => window.print());
}

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  fetchCandidates();
  setupSearchAndFilters();
  setupSpeechRecognition();
  setupAccessibilityControls();
  setupEventListeners();
});
