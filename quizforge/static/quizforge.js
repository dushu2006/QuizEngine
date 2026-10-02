(() => {
  'use strict';

  const initialNode = document.getElementById('initial-state');
  const content = document.getElementById('quiz-content');
  const frame = document.getElementById('quiz-frame');
  const toast = document.getElementById('toast');
  const serverInitialState = initialNode ? JSON.parse(initialNode.textContent) : null;
  const CACHE_KEY = 'quizengine.local-session.v1';
  let savedClientSession = null;
  try {
    const raw = window.sessionStorage.getItem(CACHE_KEY);
    const parsed = raw ? JSON.parse(raw) : null;
    if (parsed?.state?.screen && Array.isArray(parsed.answers)) savedClientSession = parsed;
  } catch (_) {
    // Storage may be disabled; the live server session remains authoritative.
  }
  const resumeAfterServerRestart = Boolean(savedClientSession &&
    savedClientSession.session_epoch && serverInitialState?.session_epoch &&
    savedClientSession.session_epoch !== serverInitialState.session_epoch);
  let state = resumeAfterServerRestart ? savedClientSession.state : serverInitialState;
  const answerLedger = new Map((savedClientSession?.answers || []).map((row) => [Number(row.question_index), row]));
  // Preferences are independent of per-question/server navigation state. The
  // API remains their durable source; question responses cannot overwrite them.
  const preferences = {
    theme: state?.settings?.theme || 'light',
    zoom: Number(state?.settings?.zoom || 1),
    layout: state?.settings?.layout || null,
    chaos: [...(state?.settings?.chaos || [])]
  };
  let navigationState = state?.navigation_state || 'NOT_READY';
  let pending = false;
  let toastTimer = null;
  let autoTimer = null;
  let autoQuestionId = null;
  let lastNotice = '';

  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[char]);
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  function rememberCurrentAnswer(current = state) {
    if (!current?.answered || current.done) return;
    const index = Number(current.question_state?.current_question ?? current.screen?.question?.index);
    const selected = Number(current.selected ?? current.question_state?.selected_option);
    const option = current.screen?.options?.find((item) => Number(item.index) === selected);
    if (Number.isInteger(index) && option) answerLedger.set(index, {question_index: index, selected_text: option.text});
  }

  function clientSnapshot() {
    rememberCurrentAnswer();
    return {
      session_epoch: state?.session_epoch,
      state,
      answers: Array.from(answerLedger.values()).sort((a, b) => a.question_index - b.question_index),
      dismissed_questions: state?.session_recovery?.dismissed_questions || [],
      settled_chaos: state?.session_recovery?.settled_chaos || []
    };
  }

  function saveClientSnapshot() {
    if (!state?.session_epoch) return;
    try { window.sessionStorage.setItem(CACHE_KEY, JSON.stringify(clientSnapshot())); }
    catch (_) { /* Keep the in-memory UI functional when storage is unavailable. */ }
  }

  async function post(url, payload = {}) {
    const omitResume = url === '/api/session/resume';
    const context = state && !omitResume ? {
      expected_question_id: questionId(),
      client_snapshot: clientSnapshot()
    } : {};
    const body = {...context, ...payload};
    const response = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      credentials: 'same-origin',
      cache: 'no-store',
      body: JSON.stringify(body)
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
    return data;
  }

  async function getSession() {
    const response = await fetch('/api/session', {credentials: 'same-origin', cache: 'no-store'});
    if (!response.ok) throw new Error(`Could not resync session (${response.status})`);
    return response.json();
  }

  function notify(message) {
    if (!message) return;
    toast.textContent = message;
    toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove('show'), 2600);
  }

  function syncPreferencesFrom(next) {
    preferences.theme = next.settings.theme;
    preferences.zoom = Number(next.settings.zoom || 1);
    preferences.layout = next.settings.layout || null;
    preferences.chaos = [...(next.settings.chaos || [])];
  }

  function installState(next, {preferencesChanged = false, keepTransition = false} = {}) {
    if (preferencesChanged) syncPreferencesFrom(next);
    else {
      // A question/navigation response must never reset theme, zoom, layout,
      // or chaos even if a future API serializer omits those fields.
      next.settings = {
        ...next.settings,
        theme: preferences.theme,
        zoom: preferences.zoom,
        layout: preferences.layout,
        chaos: [...preferences.chaos]
      };
    }
    state = next;
    rememberCurrentAnswer(state);
    if (!keepTransition && state.navigation_state !== 'TRANSITIONING') {
      navigationState = state.navigation_state || (state.done ? 'COMPLETED' : (state.answered ? 'READY' : 'NOT_READY'));
    }
    render();
    saveClientSnapshot();
  }

  async function request(url, payload = {}, {preferencesChanged = false} = {}) {
    if (pending) return null;
    pending = true;
    render();
    try {
      const next = await post(url, payload);
      if (url === '/api/restart' || payload.full_matrix === true) answerLedger.clear();
      installState(next, {preferencesChanged});
      return next;
    } catch (error) {
      notify(error.message || 'Something went wrong.');
      try {
        installState(await getSession());
      } catch (_) {
        // Keep the last known, internally consistent state if resync also fails.
      }
      return null;
    } finally {
      pending = false;
      render();
    }
  }

  function questionId(current = state) {
    return current?.question_state?.question_id ?? current?.screen?.question?.id ?? null;
  }

  function assertQuestionCommitted(expected) {
    if (expected.done) {
      if (expected.screen?.screen_role !== 'end_state' || !content.querySelector('.end-state')) {
        throw new Error('The completion screen was not rendered.');
      }
      return;
    }
    const question = content.querySelector('.question-block');
    const visibleText = question?.querySelector('h2')?.textContent?.trim();
    const expectedText = String(expected.screen?.question?.text ?? '').trim();
    const expectedId = questionId(expected);
    if (!question || !expectedId || question.dataset.questionId !== expectedId ||
        visibleText !== expectedText || frame.dataset.questionId !== expectedId) {
      throw new Error('The returned question was not committed to the visible UI.');
    }
  }

  function isBlockingOverlay() {
    return Boolean(state?.screen?.overlays?.some((overlay) => overlay.kind === 'modal'));
  }

  function scheduleAutoAdvance() {
    const automatic = Boolean(state?.answered && state?.screen?.navigation?.auto_advance && !state.done);
    const id = questionId();
    if (!automatic || pending || !id || autoQuestionId === id) return;
    clearTimeout(autoTimer);
    autoQuestionId = id;
    autoTimer = setTimeout(() => {
      autoTimer = null;
      if (pending) {
        autoQuestionId = null;
        scheduleAutoAdvance();
      } else if (state?.answered && questionId() === id) {
        autoQuestionId = null;
        navigate('next');
      }
    }, 650);
  }

  function updateControlState() {
    const themeButtons = document.querySelectorAll('[data-theme-choice]');
    themeButtons.forEach((button) => {
      const active = button.dataset.themeChoice === preferences.theme;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
      button.disabled = pending;
    });
    document.querySelectorAll('[data-zoom]').forEach((button) => {
      const active = Number(button.dataset.zoom) === preferences.zoom;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
      button.disabled = pending;
    });
    const variantSelect = document.getElementById('variant-select');
    const selectedVariant = preferences.layout || state?.screen?.variant;
    if (variantSelect && selectedVariant && Array.from(variantSelect.options).some((option) => option.value === selectedVariant)) {
      variantSelect.value = selectedVariant;
    }
    if (variantSelect) variantSelect.disabled = pending;
    document.querySelectorAll('.chaos-options input').forEach((box) => {
      box.checked = preferences.chaos.includes(box.value);
      box.disabled = pending;
    });
    const applyChaos = document.getElementById('apply-chaos');
    if (applyChaos) applyChaos.disabled = pending;
    const restart = document.getElementById('restart-button');
    if (restart) restart.disabled = pending;
    document.getElementById('chaos-status').textContent = preferences.chaos.length
      ? `${preferences.chaos.length} condition${preferences.chaos.length === 1 ? '' : 's'} active`
      : 'Conditions are off';
  }

  function render() {
    if (!state) return;
    const screen = state.screen;
    navigationState = navigationState === 'TRANSITIONING'
      ? navigationState
      : (state.navigation_state || (state.done ? 'COMPLETED' : (state.answered ? 'READY' : 'NOT_READY')));

    // One theme source of truth, applied at the document shell (not question DOM).
    document.documentElement.dataset.theme = preferences.theme;
    document.body.classList.toggle('theme-dark', preferences.theme === 'dark');
    document.documentElement.style.colorScheme = preferences.theme;
    const themeColor = document.querySelector('meta[name="theme-color"]');
    if (themeColor) themeColor.content = preferences.theme === 'dark' ? '#1a211c' : '#f5f6f3';
    frame.classList.toggle('low-contrast', Boolean(screen.chaos?.low_contrast));
    frame.classList.toggle('layout-shift', Boolean(screen.chaos?.layout_shift));
    frame.style.setProperty('--screen-zoom', String(preferences.zoom));
    frame.dataset.navigationState = navigationState;
    frame.dataset.questionId = questionId() || '';
    frame.setAttribute('aria-busy', String(pending || navigationState === 'TRANSITIONING'));
    content.setAttribute('aria-busy', String(pending || navigationState === 'TRANSITIONING'));

    const progress = document.getElementById('progress-fill');
    const total = state.question_count || 1;
    const current = screen.screen_role === 'end_state' ? total : (screen.question.index + 1);
    progress.style.width = `${Math.min(100, Math.max(0, current / total * 100))}%`;
    document.getElementById('progress-label').textContent = screen.screen_role === 'end_state' ? 'QUIZ COMPLETE' : `QUESTION ${String(current).padStart(2, '0')}`;
    document.getElementById('progress-fraction').textContent = `${current} / ${total}`;
    document.getElementById('variant-chip').textContent = screen.variant || 'results';
    updateControlState();

    if (state.notice && state.notice !== lastNotice) notify(state.notice);
    lastNotice = state.notice || '';

    if (screen.screen_role === 'end_state') {
      const results = screen.results || {};
      content.innerHTML = `
        <div class="end-state">
          <div class="result-mark" aria-hidden="true">✓</div>
          <div class="question-kicker">PRACTICE SESSION COMPLETE</div>
          <h2>${esc(results.headline || 'Quiz complete')}</h2>
          <p>${esc(results.footer || 'Thanks for taking the quiz.')}</p>
          <div class="result-score">${esc(results.score || `${state.correct_count} correct`)}</div>
          <p class="result-sub">Your answers were checked locally in this browser.</p>
          <div class="result-actions"><button type="button" class="primary-button" data-action="restart">Practice again <span class="arrow">↻</span></button><a class="button" href="/api/answer-key?format=csv">Export answer key ↓</a></div>
        </div>`;
      scheduleAutoAdvance();
      return;
    }

    const style = `style-${screen.style}`;
    const columns = Math.max(1, Number(screen.grid_columns || screen.columns || 1));
    const classes = `options ${screen.layout_class || 'layout-vertical'} ${style}`;
    const options = screen.options.map((option) => {
      const selected = state.selected === option.index;
      let resultClass = '';
      if (state.feedback && selected) resultClass = state.feedback.correct ? 'correct' : 'incorrect';
      if (state.feedback && option.index === Number(state.feedback.correct_letter.charCodeAt(0) - 65)) resultClass = 'correct';
      const disabled = state.answered || pending || isBlockingOverlay();
      return `<button type="button" class="option-button ${selected ? 'selected' : ''} ${resultClass}" data-action="answer" data-option="${option.index}" ${disabled ? 'disabled' : ''} aria-pressed="${selected}" aria-label="${esc(option.letter)}. ${esc(option.text)}"><span class="option-letter" aria-hidden="true">${esc(option.letter)}</span><span>${esc(option.text)}</span></button>`;
    }).join('');

    const feedback = state.feedback ? `<div class="feedback-box ${state.feedback.correct ? 'correct-feedback' : 'incorrect-feedback'}" role="status"><span class="feedback-symbol" aria-hidden="true">${state.feedback.correct ? '✓' : '↗'}</span><span><strong>${state.feedback.correct ? 'That’s right.' : `The correct answer is ${esc(state.feedback.correct_letter)}.`}</strong>${state.feedback.correct ? ' Your selection has been recorded.' : ` ${esc(state.feedback.correct_text)}`}</span></div>` : '';
    const answered = state.answered;
    const auto = Boolean(screen.navigation?.auto_advance);
    const canPrevious = Number(screen.question.index) > 0 && Boolean(screen.navigation?.prev_label);
    const nextText = screen.navigation?.next_label || 'Next';
    const statusText = navigationState === 'TRANSITIONING'
      ? 'Changing question…'
      : (auto && answered ? 'Answer recorded · advancing automatically…' : (answered ? 'Answer recorded' : 'Select one answer to continue'));
    const previous = canPrevious
      ? `<button type="button" class="button button-quiet previous-button" data-action="previous" ${pending ? 'disabled' : ''}>← Previous</button>`
      : '<span class="previous-placeholder" aria-hidden="true"></span>';
    const next = auto
      ? '<span class="auto-advance-state" role="status">Auto-advance</span>'
      : `<button type="button" id="continue-button" class="primary-button" data-action="next" ${(!answered || pending || navigationState === 'TRANSITIONING') ? 'disabled' : ''}>${answered ? esc(nextText) : 'Choose an answer'} <span class="arrow" aria-hidden="true">→</span></button>`;

    content.innerHTML = `
      ${screen.overlays.some((overlay) => overlay.kind === 'toast') ? '<div class="toast-banner" role="status">✓ Saved</div>' : ''}
      ${screen.chaos?.stale ? '<div class="stale-banner"><button type="button" data-action="refresh-stale" class="stale-refresh">A frame is taking longer than expected · Refresh</button></div>' : ''}
      <div class="question-block" data-question-id="${esc(questionId())}"><div class="question-kicker">${screen.question.progress_text ? esc(screen.question.progress_text.toUpperCase()) : 'TAKE YOUR TIME'}</div><h2>${esc(screen.question.text)}</h2></div>
      <div class="${classes}" style="--grid-columns:${columns}" role="group" aria-label="Answer choices">${options}</div>
      ${feedback}
      <div class="quiz-actions">${previous}<span class="action-hint" role="status" aria-live="polite">${statusText}</span>${next}</div>
      ${screen.overlays.some((overlay) => overlay.kind === 'modal') ? `<div class="overlay-scrim" role="dialog" aria-modal="true" aria-labelledby="modal-heading"><div class="modal-card"><div class="modal-top"><div class="question-kicker">SESSION NOTICE</div><button type="button" class="modal-close" data-action="dismiss-modal" aria-label="Close notice">×</button></div><h3 id="modal-heading">A quick note</h3><p>${esc(screen.overlays.find((overlay) => overlay.kind === 'modal').text)}</p><button type="button" class="primary-button" data-action="dismiss-modal">Got it <span class="arrow" aria-hidden="true">✓</span></button></div></div>` : ''}`;

    scheduleAutoAdvance();
  }

  async function choose(index) {
    if (pending || !state || state.answered || isBlockingOverlay()) return;
    clearTimeout(autoTimer);
    autoTimer = null;
    autoQuestionId = null;
    await request('/api/answer', {option: index});
  }

  async function navigate(direction) {
    if (pending || !state || navigationState === 'TRANSITIONING') return;
    if (direction === 'next' && (!state.answered || state.done)) return;
    if (direction === 'previous' && Number(state.screen.question.index) <= 0) return;

    const oldIndex = Number(state.screen.question.index);
    const oldId = questionId();
    const oldText = String(state.screen.question.text ?? '').trim();
    if (!oldId) {
      notify('Cannot navigate: the current question has no identity.');
      return;
    }
    pending = true;
    navigationState = 'TRANSITIONING';
    render();
    try {
      if (direction === 'next') {
        if (preferences.chaos.includes('delay')) {
          notify('Waiting for the next screen…');
          await sleep(850);
        }
        if (preferences.chaos.includes('layout_shift') || preferences.chaos.includes('stale')) {
          // Resolve the visual-condition fixture before changing question. Keep
          // the in-flight lock throughout so fast/double clicks cannot race it.
          const settled = await post('/api/chaos/advance');
          if (questionId(settled) !== oldId || Number(settled.screen?.question?.index) !== oldIndex) {
            throw new Error('The current question changed while settling its screen condition.');
          }
          installState(settled, {keepTransition: true});
          assertQuestionCommitted(settled);
        }
      }
      const next = await post(direction === 'previous' ? '/api/previous' : '/api/next');
      const newIndex = Number(next.screen?.question?.index);
      const newId = questionId(next);
      if (next.success !== true || next.from_question_id !== oldId) {
        throw new Error('Navigation response does not match the question we left.');
      }
      if (next.done) {
        if (direction !== 'next' || oldIndex !== Number(next.total_questions) - 1 ||
            next.navigation_state !== 'COMPLETED' || next.question_id !== null) {
          throw new Error('The final-question response did not confirm completion.');
        }
      } else {
        const expectedIndex = direction === 'previous' ? oldIndex - 1 : oldIndex + 1;
        if (newIndex !== expectedIndex || next.question_number !== expectedIndex + 1) {
          throw new Error('Navigation did not advance exactly one question.');
        }
        if (!newId || next.question_id !== newId || next.screen.question.id !== newId || newId === oldId) {
          throw new Error('Navigation did not return a new stable question identity.');
        }
        if (direction === 'next' && String(next.question ?? '').trim() === oldText) {
          throw new Error('Next returned the previous question content.');
        }
        if (String(next.question ?? '') !== String(next.screen.question.text ?? '') ||
            JSON.stringify(next.options) !== JSON.stringify(next.screen.options)) {
          throw new Error('Navigation response content does not match its rendered screen.');
        }
      }

      // Keep TRANSITIONING set while the new screen is synchronously installed
      // and verified in the DOM. Unlocking happens in finally only after this.
      installState(next, {keepTransition: true});
      assertQuestionCommitted(next);
      navigationState = next.done ? 'COMPLETED' : next.navigation_state;
      clearTimeout(autoTimer);
      autoTimer = null;
      autoQuestionId = null;
    } catch (error) {
      notify(error.message || 'Navigation failed.');
      // Never leave a possibly stale response on screen: read an uncached
      // snapshot from the serialized server state and commit that exact state.
      try {
        const latest = await getSession();
        const latestIndex = Number(latest?.question_state?.current_question);
        const latestId = questionId(latest);
        const expectedIndex = direction === 'previous' ? oldIndex - 1 : oldIndex + 1;
        const expectedCommit = latest.done
          ? direction === 'next' && oldIndex === Number(latest.question_count) - 1
          : latestIndex === expectedIndex && latestId !== oldId &&
            (direction !== 'next' || String(latest.screen.question.text ?? '').trim() !== oldText);
        if (expectedCommit) {
          installState(latest, {keepTransition: true});
          assertQuestionCommitted(latest);
          navigationState = latest.navigation_state || (latest.done ? 'COMPLETED' : (latest.answered ? 'READY' : 'NOT_READY'));
        } else {
          // Never turn a stale 409/resync into a silent rewind to Q1. Keep the
          // user's last known rendered/answered state and explain recovery.
          if (latestIndex < oldIndex || (latestIndex === oldIndex && latestId === oldId)) {
            notify('The server session was reset; this screen was kept instead of silently restarting the quiz.');
          }
        }
      } catch (_) {
        // Keep the last known screen and the lock until the transition is over.
      }
    } finally {
      pending = false;
      navigationState = state?.done ? 'COMPLETED' : (state?.answered ? 'READY' : 'NOT_READY');
      render();
    }
  }

  // One delegated listener survives every content.innerHTML replacement; no
  // controls retain stale nodes or closures after question renders.
  content.addEventListener('click', (event) => {
    const button = event.target.closest('[data-action]');
    if (!button || !content.contains(button) || pending || button.disabled) return;
    event.preventDefault();
    const action = button.dataset.action;
    if (action === 'answer') choose(Number(button.dataset.option));
    else if (action === 'next') navigate('next');
    else if (action === 'previous') navigate('previous');
    else if (action === 'restart') request('/api/restart');
    else if (action === 'dismiss-modal') request('/api/overlay/dismiss');
    else if (action === 'refresh-stale') request('/api/chaos/advance');
  });

  document.getElementById('theme-control').addEventListener('click', (event) => {
    const button = event.target.closest('[data-theme-choice]');
    if (button && !pending) request('/api/settings', {theme: button.dataset.themeChoice}, {preferencesChanged: true});
  });
  document.getElementById('zoom-control').addEventListener('click', (event) => {
    const button = event.target.closest('[data-zoom]');
    if (button && !pending) request('/api/settings', {zoom: Number(button.dataset.zoom)}, {preferencesChanged: true});
  });
  document.getElementById('variant-select').addEventListener('change', (event) => {
    if (!pending) request('/api/settings', {variant: event.target.value}, {preferencesChanged: true});
  });
  document.getElementById('apply-chaos').addEventListener('click', async () => {
    if (pending) return;
    const chaos = Array.from(document.querySelectorAll('.chaos-options input:checked')).map((input) => input.value);
    const next = await request('/api/settings', {chaos}, {preferencesChanged: true});
    if (next) document.querySelector('.chaos-panel').open = false;
  });
  document.getElementById('restart-button').addEventListener('click', () => {
    if (!pending) request('/api/restart');
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && isBlockingOverlay() && !pending) request('/api/overlay/dismiss');
  });

  // Keep the fixture sweep as an explicit, functional utility control.
  const fullSweep = document.createElement('button');
  fullSweep.className = 'matrix-link';
  fullSweep.type = 'button';
  fullSweep.textContent = 'Run all 18 quiz layouts →';
  fullSweep.setAttribute('aria-label', 'Start a full sweep through all eighteen answerable layouts');
  document.querySelector('.workbench-heading').appendChild(fullSweep);
  fullSweep.addEventListener('click', async () => {
    if (pending) return;
    const next = await request('/api/settings', {full_matrix: true}, {preferencesChanged: true});
    if (next) notify('Full sweep through 18 quiz layouts started.');
  });

  render();
  rememberCurrentAnswer(state);
  saveClientSnapshot();
  if (resumeAfterServerRestart) {
    request('/api/session/resume', {snapshot: savedClientSession}, {preferencesChanged: true});
  }
})();
