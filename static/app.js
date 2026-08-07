// AI Technical Interview Agent Client App Logic
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
  selectedModel: 'gemma-4-31b-it'
};

// DOM Elements
const views = {
  candidateSelect: document.getElementById('view-candidate-select'),
  interviewRoom: document.getElementById('view-interview-room'),
  feedbackPortal: document.getElementById('view-feedback-portal')
};

// Announcement Helper for Screen Readers
function announceSR(message) {
  const container = document.getElementById('sr-announcements');
  if (container) {
    container.textContent = '';
    setTimeout(() => {
      container.textContent = message;
    }, 50);
  }
}

// Switch Views
function showView(viewName) {
  Object.keys(views).forEach(v => {
    if (v === viewName) {
      views[v].classList.add('active');
      views[v].focus();
    } else {
      views[v].classList.remove('active');
    }
  });
}

// Fetch Candidates on Load
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
      container.innerHTML = `<p class="error-msg">Failed to load cohort candidate profiles. Check backend connection.</p>`;
    }
  }
}

// Render Candidate Cards
function renderCandidates(candidatesList) {
  const container = document.getElementById('candidates-grid');
  if (!container) return;

  if (candidatesList.length === 0) {
    container.innerHTML = `<p>No matching candidate profiles found.</p>`;
    return;
  }

  container.innerHTML = candidatesList.map(c => {
    const mem = c.member;
    const sig = c.signals || {};
    const firstTryPercent = Math.round((sig.missionsFirstTry / (sig.missionsCompleted || 1)) * 100);

    return `
      <article class="candidate-card" role="listitem" tabindex="0" aria-label="${mem.name}, ${mem.jobRole}">
        <div class="candidate-card-header">
          <div>
            <h3 class="cand-name">${mem.name}</h3>
            <span class="cand-role">${mem.jobRole}</span>
          </div>
          <span class="badge-role">${mem.yearsExperience} Yrs Exp</span>
        </div>

        <ul class="cand-stats-list" aria-label="Learning Journey Metrics">
          <li><span>Missions Completed:</span> <strong>${sig.missionsCompleted} / 31 Days</strong></li>
          <li><span>First-Try Pass Rate:</span> <strong>${firstTryPercent}%</strong></li>
          <li><span>Commit Frequency:</span> <strong>${sig.commitDays} Days</strong></li>
          <li><span>Education:</span> <strong>${mem.education}</strong></li>
        </ul>

        <button class="btn-start-interview" data-id="${mem.id}" aria-label="Start Technical Interview for ${mem.name}">
          Start Technical Interview ➔
        </button>
      </article>
    `;
  }).join('');

  // Attach Click Listeners
  container.querySelectorAll('.btn-start-interview').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const candId = e.currentTarget.getAttribute('data-id');
      startInterviewSession(candId);
    });
  });
}

// Filter Candidates
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
    });
  }

  filterPills.forEach(pill => {
    pill.addEventListener('click', (e) => {
      filterPills.forEach(p => p.classList.remove('active'));
      e.target.classList.add('active');

      const filter = e.target.getAttribute('data-filter');
      let filtered = state.candidates;

      if (filter === 'senior') {
        filtered = state.candidates.filter(c => c.member.yearsExperience >= 5);
      } else if (filter === 'junior') {
        filtered = state.candidates.filter(c => c.member.yearsExperience < 5);
      } else if (filter === 'perfect') {
        filtered = state.candidates.filter(c => (c.signals?.missionsCompleted || 0) >= 30);
      }

      renderCandidates(filtered);
    });
  });
}

// Start Interview Session
async function startInterviewSession(candidateId) {
  try {
    const res = await fetch('/api/interview/start', {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'X-GEMINI-API-KEY': state.byokApiKey
      },
      body: JSON.stringify({ candidate_id: candidateId })
    });

    const data = await res.json();
    state.sessionId = data.session_id;
    state.selectedCandidate = data.candidate;
    state.questionsAsked = data.questions_asked;
    state.daysCovered = data.days_covered_list;

    // Update UI Header
    document.getElementById('heading-interview-room').textContent = `${data.candidate.name}'s Interview`;
    document.getElementById('candidate-role-badge').textContent = data.candidate.jobRole;

    updateProgressGauges(data.questions_asked, data.days_covered_list, data.current_topic);

    // Clear and Append First Question
    const chatLog = document.getElementById('chat-messages-log');
    chatLog.innerHTML = '';
    appendMessage('agent', data.initial_question);

    showView('interviewRoom');
    announceSR(`Interview session started for ${data.candidate.name}. First question asked.`);

    if (state.ttsEnabled) {
      speakText(data.initial_question);
    }
  } catch (err) {
    console.error('Failed to start interview:', err);
    alert('Error starting interview session. Check backend.');
  }
}

// Update Progress Counters
function updateProgressGauges(questionsCount, daysList, currentTopic) {
  document.getElementById('gauge-questions-count').textContent = `Q${questionsCount} / 8+`;
  document.getElementById('gauge-days-count').textContent = `${daysList.length} / 4+ Days`;

  if (currentTopic) {
    document.getElementById('current-topic-badge').textContent = currentTopic;
  }

  const container = document.getElementById('days-badges-container');
  if (container) {
    container.innerHTML = daysList.map(d => `<span class="day-chip">Day ${d}</span>`).join('');
  }
}

// Append Chat Message
function appendMessage(role, text) {
  const chatLog = document.getElementById('chat-messages-log');
  if (!chatLog) return;

  const bubble = document.createElement('div');
  bubble.className = `chat-bubble ${role}`;
  bubble.setAttribute('tabindex', '0');

  const meta = document.createElement('div');
  meta.className = 'bubble-meta';
  meta.textContent = role === 'agent' ? '🤖 AI Lead Architect' : '👤 Candidate Response';

  const content = document.createElement('div');
  content.className = 'bubble-text';
  content.innerHTML = text.replace(/\n/g, '<br/>');

  bubble.appendChild(meta);
  bubble.appendChild(content);
  chatLog.appendChild(bubble);

  chatLog.scrollTop = chatLog.scrollHeight;
}

// Submit Answer
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
        'X-GEMINI-API-KEY': state.byokApiKey
      },
      body: JSON.stringify({
        session_id: state.sessionId,
        message: text
      })
    });

    const data = await res.json();
    state.questionsAsked = data.questions_asked;
    state.daysCovered = data.days_covered_list;

    updateProgressGauges(data.questions_asked, data.days_covered_list);
    appendMessage('agent', data.agent_response);

    if (state.ttsEnabled) {
      speakText(data.agent_response);
    }

    if (data.is_complete) {
      setTimeout(() => fetchFeedbackReport(), 1500);
    }
  } catch (err) {
    console.error('Error submitting answer:', err);
  }
}

// Speech Synthesis (TTS)
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

// Speech Recognition (STT)
function setupSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    console.warn('Speech recognition not supported in this browser.');
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = true;

  const micBtn = document.getElementById('btn-toggle-mic');
  const statusBar = document.getElementById('stt-status-bar');
  const inputText = document.getElementById('candidate-input-text');

  micBtn.addEventListener('click', () => {
    if (state.isListening) {
      recognition.stop();
    } else {
      recognition.start();
    }
  });

  recognition.onstart = () => {
    state.isListening = true;
    micBtn.classList.add('listening');
    statusBar.classList.remove('hidden');
    announceSR('Microphone active. Listening for candidate response.');
  };

  recognition.onresult = (event) => {
    const transcript = Array.from(event.results)
      .map(result => result[0].transcript)
      .join('');
    inputText.value = transcript;
  };

  recognition.onend = () => {
    state.isListening = false;
    micBtn.classList.remove('listening');
    statusBar.classList.add('hidden');
  };
}

// Fetch Final Feedback Report
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
  }
}

// Render Feedback Report
function renderFeedbackReport(report) {
  document.getElementById('report-candidate-subtitle').textContent = `Candidate: ${report.candidate.name} | ${report.candidate.jobRole}`;
  document.getElementById('report-overall-score').textContent = report.overall_score;
  document.getElementById('report-readiness-level').textContent = report.readiness_level;
  document.getElementById('report-recommendation-text').textContent = report.recommendation;

  document.getElementById('report-questions-total').textContent = `${report.questions_asked_total} Questions`;
  document.getElementById('report-days-total').textContent = `${report.unique_days_covered_count} Days`;

  // Render Domain Table (Accessible)
  const tbody = document.getElementById('domain-table-body');
  if (tbody && report.domain_scores) {
    tbody.innerHTML = Object.entries(report.domain_scores).map(([domain, score]) => `
      <tr>
        <th scope="row">${domain}</th>
        <td><strong>${score} / 100</strong></td>
        <td><span class="score-tag ${score >= 80 ? 'pass' : ''}">${score >= 80 ? 'MASTERY' : 'REVIEW'}</span></td>
      </tr>
    `).join('');
  }

  // Strengths & Growth Areas
  const strengthsList = document.getElementById('report-strengths-list');
  if (strengthsList && report.strengths) {
    strengthsList.innerHTML = report.strengths.map(s => `<li>${s}</li>`).join('');
  }

  const growthList = document.getElementById('report-growth-list');
  if (growthList && report.growth_areas) {
    growthList.innerHTML = report.growth_areas.map(g => `<li>${g}</li>`).join('');
  }

  // Recommended Review Days
  const reviewContainer = document.getElementById('report-review-days-container');
  if (reviewContainer && report.recommended_review_days) {
    reviewContainer.innerHTML = report.recommended_review_days.map(day => `
      <div class="card-box" style="padding: 1rem; margin-bottom: 0;">
        <span class="badge-role">Day ${day}</span>
        <h4 style="margin-top: 0.5rem;">Targeted Curriculum Day ${day}</h4>
        <p style="font-size: 0.85rem; color: var(--text-muted);">Revisit concepts and hands-on exercises for Day ${day}.</p>
      </div>
    `).join('');
  }
}

// Setup Keyboard Hotkeys & Accessibility Controls
function setupAccessibilityControls() {
  const fontInc = document.getElementById('btn-font-inc');
  const fontDec = document.getElementById('btn-font-dec');
  const contrastBtn = document.getElementById('btn-toggle-contrast');
  const ttsBtn = document.getElementById('btn-toggle-tts');
  const rateSelect = document.getElementById('speech-rate-select');

  if (fontInc) {
    fontInc.addEventListener('click', () => {
      document.documentElement.setAttribute('data-size', 'large');
      document.getElementById('font-size-indicator').textContent = '125%';
    });
  }
  if (fontDec) {
    fontDec.addEventListener('click', () => {
      document.documentElement.setAttribute('data-size', 'normal');
      document.getElementById('font-size-indicator').textContent = '100%';
    });
  }
  if (contrastBtn) {
    contrastBtn.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-contrast');
      const next = current === 'high' ? 'normal' : 'high';
      document.documentElement.setAttribute('data-contrast', next);
      contrastBtn.classList.toggle('active', next === 'high');
      announceSR(`High contrast mode set to ${next}`);
    });
  }
  if (ttsBtn) {
    ttsBtn.addEventListener('click', () => {
      state.ttsEnabled = !state.ttsEnabled;
      ttsBtn.classList.toggle('active', state.ttsEnabled);
      ttsBtn.querySelector('.label').textContent = `Voice TTS: ${state.ttsEnabled ? 'ON' : 'OFF'}`;
    });
  }
  if (rateSelect) {
    rateSelect.addEventListener('change', (e) => {
      state.speechRate = parseFloat(e.target.value);
    });
  }

  // Keyboard Hotkeys
  document.addEventListener('keydown', (e) => {
    if (e.altKey && e.key.toLowerCase() === 's') {
      e.preventDefault();
      document.getElementById('btn-toggle-mic').click();
    } else if (e.altKey && e.key.toLowerCase() === 'r') {
      e.preventDefault();
      const agentMsgs = document.querySelectorAll('.chat-bubble.agent');
      if (agentMsgs.length > 0) {
        speakText(agentMsgs[agentMsgs.length - 1].textContent);
      }
    } else if (e.key === 'Enter' && (e.ctrlKey || document.activeElement.id === 'candidate-input-text')) {
      if (!e.shiftKey) {
        e.preventDefault();
        submitAnswer();
      }
    } else if (e.altKey && e.key.toLowerCase() === 'f') {
      e.preventDefault();
      fetchFeedbackReport();
    } else if (e.key === 'Escape') {
      document.querySelectorAll('.modal-backdrop').forEach(m => m.classList.add('hidden'));
    }
  });

  // Modal Listeners
  const helpBtn = document.getElementById('btn-keyboard-help');
  const helpModal = document.getElementById('modal-keyboard-help');
  const closeModal = document.getElementById('btn-close-modal');

  if (helpBtn && helpModal) {
    helpBtn.addEventListener('click', () => helpModal.classList.remove('hidden'));
  }
  if (closeModal && helpModal) {
    closeModal.addEventListener('click', () => helpModal.classList.add('hidden'));
  }
}

// Event Listeners for Buttons
function setupEventListeners() {
  document.getElementById('btn-submit-answer')?.addEventListener('click', submitAnswer);
  document.getElementById('btn-speak-question')?.addEventListener('click', () => {
    const agentMsgs = document.querySelectorAll('.chat-bubble.agent');
    if (agentMsgs.length > 0) {
      speakText(agentMsgs[agentMsgs.length - 1].textContent);
    }
  });
  document.getElementById('btn-back-to-list')?.addEventListener('click', () => showView('candidateSelect'));
  document.getElementById('btn-force-finish')?.addEventListener('click', fetchFeedbackReport);
  document.getElementById('btn-restart-new')?.addEventListener('click', () => showView('candidateSelect'));
  document.getElementById('btn-print-report')?.addEventListener('click', () => window.print());
}

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  fetchCandidates();
  setupSearchAndFilters();
  setupSpeechRecognition();
  setupAccessibilityControls();
  setupEventListeners();
});
