/* Neurovox Stage 4 — Mission-Control / Telemetry Frontend
 * Reads live from /api/info, /api/results and /api/verify. Fully offline. */
(function () {
  'use strict';

  /* =====================================================================
   * Helpers
   * ===================================================================== */
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const NF = new Intl.NumberFormat('en-US'); // international grouping: 1,234,567
  const isNil = (v) => v === null || v === undefined || v === '' || Number.isNaN(Number(v));
  const fmt = (n) => (isNil(n) ? '—' : NF.format(Number(n)));
  const num = (n, d = 2) => (isNil(n) ? '—' : Number(n).toFixed(d));
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const reduceMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  /* Strip markdown / LaTeX leftovers from docs-derived API strings. */
  function clean(s) {
    return String(s == null ? '' : s)
      .replace(/\$([^$]*)\$/g, '$1')
      .replace(/\\text\{([^}]*)\}/g, '$1')
      .replace(/\\mathbb\{R\}/g, 'R')
      .replace(/\\times/g, '×').replace(/\\to/g, '→').replace(/\\in/g, '∈')
      .replace(/\\_/g, '_').replace(/\\\\/g, '')
      .replace(/\\([a-zA-Z]+)/g, '$1')
      .replace(/[`{}]/g, '').replace(/\*\*/g, '')
      .replace(/<br\s*\/?>/gi, ' ').trim();
  }

  /* Glossary (hover / focus tooltips, one sentence each). */
  const GLOSSARY = {
    'eer': 'Equal Error Rate: the point where wrongly accepting an impostor and wrongly rejecting the real speaker happen equally often; lower is better.',
    'mindcf': 'Minimum Detection Cost Function: one number that weighs missed matches against false alarms at a chosen operating point; lower is better.',
    'adaptivedrop': 'A training safeguard that ignores or relabels clips the model finds suspiciously hard, so likely-mislabeled data does less harm.',
    'ska': 'Selective Kernel Attention: lets the network weigh filters of different sizes and favor the ones that suit each sound.',
    'tdnn': 'Time-Delay Neural Network: layers that look at several neighboring moments of audio at once to capture how a voice changes over time.',
    'embedding': 'A fixed-length list of numbers (a voiceprint) that summarizes who is speaking in a clip.',
    'cosine score': 'How closely two voiceprints point in the same direction, from -1 (opposite) to 1 (identical).',
    'threshold': 'The cosine score above which the system says "same speaker"; here it is set at the equal-error point on dev-clean.',
    'borderline zone': 'A band around the threshold where the score is too close to call, so the verdict is INCONCLUSIVE.',
    'iaamsoftmax': 'The training loss: it pulls voiceprints of the same speaker together and pushes different speakers apart by an angular margin, with AdaptiveDrop built in.',
    'aamsoftmax': 'The standard additive-angular-margin training loss, used here without AdaptiveDrop.',
    'mlasp': 'Multi-layer attentive statistics pooling: combines frame-level features into one summary for the whole clip using attention.',
    'mlfc': 'Multi-level feature combination: joins the outputs of several network stages into one feature stream.',
    'log-mel': 'A picture of the sound showing energy in 80 pitch-like frequency bands over time.',
    'dev-clean': 'A LibriSpeech subset kept apart from training; it is used to choose the best checkpoint.',
    'parameters': 'The learnable numbers (weights) inside a model; more parameters means a larger model.',
    'ecapa-tdnn': 'A widely used speaker-recognition network, used here as the baseline.',
  };
  function term(name, label) {
    return `<span class="term" tabindex="0" data-term="${esc(name)}">${esc(label || name)}</span>`;
  }
  function explain(escaped) {
    return escaped.replace(/\b(AdaptiveDrop|IAAMSoftmax|AAMSoftmax|MLASP|MLFC|minDCF|TDNN|SKA|EER)\b/g,
      (m) => term(m.toLowerCase() === 'ska' ? 'SKA' : m, m));
  }

  /* Provenance pill badges (wording preserved). */
  function tagHtml(type) {
    const t = String(type || '').toLowerCase();
    const cls = t === 'measured' ? 'tag-measured'
      : t === 'external reference' ? 'tag-ext'
        : t === 'paper' ? 'tag-paper'
          : (t === 'planned' || t.indexOf('pending') >= 0) ? 'tag-planned'
            : 'tag-neutral';
    return `<span class="tag ${cls}">${esc(type)}</span>`;
  }
  function rich(s, withTerms) {
    let h = esc(clean(s));
    if (withTerms) h = explain(h);
    return h.replace(/\[(measured|paper|external reference|planned|pending|external-unverified|engineering choice)\]/gi, (m, k) => {
      const key = k.toLowerCase();
      const cls = key === 'measured' ? 'tag-measured' : key === 'external reference' ? 'tag-ext'
        : key === 'paper' ? 'tag-paper' : (key === 'planned' || key === 'pending') ? 'tag-planned' : 'tag-neutral';
      return `<span class="tag ${cls}">${m}</span>`;
    });
  }

  /* =====================================================================
   * Glossary tooltips (hover + keyboard focus, Esc dismisses)
   * ===================================================================== */
  const tipEl = $('#tooltip');
  let tipOwner = null;
  function tipText(el) {
    if (el.dataset.tip) return el.dataset.tip;
    return GLOSSARY[String(el.dataset.term || '').toLowerCase()] || '';
  }
  function showTip(el) {
    const text = tipText(el);
    if (!text) return;
    tipOwner = el;
    tipEl.textContent = text;
    tipEl.hidden = false;
    el.setAttribute('aria-describedby', 'tooltip');
    const r = el.getBoundingClientRect();
    const tr = tipEl.getBoundingClientRect();
    let left = Math.min(Math.max(8, r.left), window.innerWidth - tr.width - 8);
    let top = r.bottom + 8;
    if (top + tr.height > window.innerHeight - 8) top = Math.max(8, r.top - tr.height - 8);
    tipEl.style.left = left + 'px';
    tipEl.style.top = top + 'px';
  }
  function hideTip() {
    if (tipOwner) tipOwner.removeAttribute('aria-describedby');
    tipOwner = null;
    tipEl.hidden = true;
  }
  const TIP_SEL = '[data-term],[data-tip]';
  document.addEventListener('mouseover', (e) => { const el = e.target.closest && e.target.closest(TIP_SEL); if (el) showTip(el); });
  document.addEventListener('mouseout', (e) => {
    const el = e.target.closest && e.target.closest(TIP_SEL);
    if (el && !el.contains(e.relatedTarget)) hideTip();
  });
  document.addEventListener('focusin', (e) => { const el = e.target.closest && e.target.closest(TIP_SEL); if (el) showTip(el); });
  document.addEventListener('focusout', (e) => { if (e.target.closest && e.target.closest(TIP_SEL)) hideTip(); });
  document.addEventListener('scroll', hideTip, true);

  /* =====================================================================
   * Modal overlay: focus trap, Esc closes, focus restored
   * ===================================================================== */
  const overlay = $('#overlay');
  const modal = $('#modal');
  const BACKGROUND_IDS = ['topbar', 'main'];
  let modalOpener = null;
  function focusables() {
    return $$('a[href],button:not([disabled]),input:not([disabled]),select,textarea,summary,[tabindex]:not([tabindex="-1"])', modal)
      .filter((el) => el.offsetParent !== null || el === document.activeElement);
  }
  function openModal(title, html, opener, eyebrow) {
    hideTip();
    modalOpener = opener || document.activeElement;
    $('#modal-eyebrow').textContent = eyebrow || 'TELEMETRY INSPECTOR';
    $('#modal-title').textContent = title;
    $('#modal-body').innerHTML = html;
    $('#modal-body').scrollTop = 0;
    overlay.hidden = false;
    BACKGROUND_IDS.forEach((id) => {
      const el = $('#' + id);
      if (el) { el.setAttribute('inert', ''); el.setAttribute('aria-hidden', 'true'); }
    });
    modal.focus();
  }
  function closeModal() {
    if (overlay.hidden) return;
    overlay.hidden = true;
    BACKGROUND_IDS.forEach((id) => {
      const el = $('#' + id);
      if (el) { el.removeAttribute('inert'); el.removeAttribute('aria-hidden'); }
    });
    if (modalOpener && typeof modalOpener.focus === 'function') modalOpener.focus();
    modalOpener = null;
  }
  $('#modal-close').addEventListener('click', closeModal);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) closeModal(); });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      if (!tipEl.hidden) hideTip();
      if (!overlay.hidden) { e.preventDefault(); closeModal(); }
      return;
    }
    if (e.key === 'Tab' && !overlay.hidden) {
      const list = focusables();
      if (!list.length) { e.preventDefault(); modal.focus(); return; }
      const first = list[0];
      const last = list[list.length - 1];
      const active = document.activeElement;
      if (e.shiftKey && (active === first || active === modal)) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && active === last) { e.preventDefault(); first.focus(); }
      else if (!modal.contains(active)) { e.preventDefault(); first.focus(); }
    }
  });

  /* =====================================================================
   * Tabs: 5 tabs (Demo, How it works, Results, Models, Rigor).
   * Old #roadmap hash redirects to #demo.
   * ===================================================================== */
  const tabButtons = $$('[role="tab"]');
  const TAB_IDS = tabButtons.map((b) => b.dataset.tab);
  let currentTab = null;
  function activateTab(id, opts) {
    const o = Object.assign({ focus: false, updateHash: true }, opts || {});
    const requestedRoadmap = id === 'roadmap';
    if (requestedRoadmap || TAB_IDS.indexOf(id) < 0) id = 'demo';
    const changed = id !== currentTab;
    currentTab = id;
    tabButtons.forEach((b) => {
      const on = b.dataset.tab === id;
      b.setAttribute('aria-selected', on ? 'true' : 'false');
      b.tabIndex = on ? 0 : -1;
      if (on && o.focus) b.focus();
    });
    $$('[role="tabpanel"]').forEach((p) => {
      const on = p.id === 'panel-' + id;
      p.hidden = !on;
      if (on && changed) { p.classList.remove('enter'); void p.offsetWidth; p.classList.add('enter'); }
    });
    if ((o.updateHash || requestedRoadmap) && location.hash !== '#' + id) {
      location.hash = id;
    }
  }
  tabButtons.forEach((b) => b.addEventListener('click', () => activateTab(b.dataset.tab)));
  $('.tabs').addEventListener('keydown', (e) => {
    const idx = TAB_IDS.indexOf(currentTab);
    let next = -1;
    if (e.key === 'ArrowRight') next = (idx + 1) % TAB_IDS.length;
    else if (e.key === 'ArrowLeft') next = (idx - 1 + TAB_IDS.length) % TAB_IDS.length;
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = TAB_IDS.length - 1;
    if (next >= 0) { e.preventDefault(); activateTab(TAB_IDS[next], { focus: true }); }
  });
  window.addEventListener('hashchange', () => activateTab(location.hash.replace('#', ''), { updateHash: false }));

  /* =====================================================================
   * Shared data + derived status (computed from history.csv / run_meta.json)
   * ===================================================================== */
  const PLANNED_EPOCHS = 30;
  const DATA = { info: null, results: null };

  const runEpochs = (run) => (run && run.epochs) || [];
  function progress(run) {
    if (!run) return null;
    const k = run.num_epochs_logged || runEpochs(run).length;
    return { k, total: Math.max(PLANNED_EPOCHS, k), done: k >= PLANNED_EPOCHS };
  }
  function adaptiveStartEpoch(run) {
    const r = runEpochs(run).find((x) => (x.loss_num_filtered || 0) > 0 || (!isNil(x.loss_keep_ratio) && x.loss_keep_ratio < 1));
    return r ? r.epoch : null;
  }
  /* S1 appears on comparison charts only when its logged epoch count is comparable to S0's. */
  function s1Comparable(s0, s1) {
    return !!(s0 && s1 && (s1.num_epochs_logged || 0) >= (s0.num_epochs_logged || 0));
  }
  function runStatusText(run, dataStatus) {
    const p = progress(run);
    if (!p) return dataStatus;
    if (p.done) return dataStatus || `trained (${p.k}/${p.total} epochs; best dev EER ${num(run.best_dev_eer_pct)}%)`;
    return `in progress (epoch ${p.k}/${p.total}; best dev EER so far ${num(run.best_dev_eer_pct)}%)`;
  }
  function switchStatus(sw, runs) {
    if (sw.switch === 'S0') return runStatusText(runs.s0, sw.status);
    if (sw.switch === 'S1') return runStatusText(runs.s1, sw.status);
    return sw.status;
  }

  /* =====================================================================
   * SVG line chart with hover / keyboard tooltips
   * ===================================================================== */
  function lineChart(host, o) {
    const W = 560, H = 220, pl = 48, pr = 14, pt = 28, pb = 26;
    host.innerHTML = '';
    const series = (o.series || []).filter((s) => s.points && s.points.length);
    if (!series.length) return;
    const xs = Array.from(new Set(series.flatMap((s) => s.points.map((p) => p.x)))).sort((a, b) => a - b);
    const ys = series.flatMap((s) => s.points.map((p) => p.y));
    let xMin = o.xMin != null ? o.xMin : xs[0];
    let xMax = o.xMax != null ? o.xMax : xs[xs.length - 1];
    if (xMax === xMin) xMax = xMin + 1;
    let yMin = o.yMin != null ? o.yMin : Math.min.apply(null, ys);
    let yMax = o.yMax != null ? o.yMax : Math.max.apply(null, ys);
    if (yMax === yMin) yMax = yMin + 1;
    const padY = (yMax - yMin) * 0.1;
    if (o.yMin == null) yMin = Math.max(Math.min.apply(null, ys) >= 0 ? 0 : -Infinity, yMin - padY);
    if (o.yMax == null) yMax += padY;
    const yFmt = o.yFmt || ((v) => v.toFixed(1));
    const sx = (x) => pl + ((x - xMin) / (xMax - xMin)) * (W - pl - pr);
    const sy = (y) => H - pb - ((y - yMin) / (yMax - yMin)) * (H - pt - pb);

    let svg = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img" tabindex="0" aria-label="${esc(o.aria || 'chart')}. Use left and right arrow keys to inspect each epoch.">`;
    for (let i = 0; i <= 4; i++) {
      const v = yMin + ((yMax - yMin) * i) / 4;
      svg += `<line class="gridline" x1="${pl}" x2="${W - pr}" y1="${sy(v).toFixed(1)}" y2="${sy(v).toFixed(1)}"/>`;
      svg += `<text class="ax" x="${pl - 6}" y="${(sy(v) + 3.5).toFixed(1)}" text-anchor="end">${esc(yFmt(v))}</text>`;
    }
    const step = Math.max(1, Math.ceil((xMax - xMin) / 10));
    for (let x = Math.ceil(xMin); x <= xMax; x += step) {
      svg += `<text class="ax" x="${sx(x).toFixed(1)}" y="${H - 7}" text-anchor="middle">${x}</text>`;
    }
    if (o.mark && o.mark.x >= xMin && o.mark.x <= xMax) {
      const mx = sx(o.mark.x);
      svg += `<line x1="${mx.toFixed(1)}" x2="${mx.toFixed(1)}" y1="${pt}" y2="${H - pb}" stroke="#fbbf24" stroke-width="1.4" stroke-dasharray="4,3"/>`;
      svg += `<text x="${(mx + 5).toFixed(1)}" y="${pt + 10}" font-family="ui-monospace, monospace" font-size="10" font-weight="700" fill="#fbbf24">${esc(o.mark.label)}</text>`;
    }
    series.forEach((s) => {
      const pts = s.points.map((p) => `${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ');
      svg += `<polyline fill="none" stroke="${s.color}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round" points="${pts}" style="filter:drop-shadow(0 0 5px ${s.color})"/>`;
      s.points.forEach((p) => { svg += `<circle cx="${sx(p.x).toFixed(1)}" cy="${sy(p.y).toFixed(1)}" r="2.5" fill="${s.color}"/>`; });
    });
    let lx = pl;
    series.forEach((s) => {
      svg += `<rect x="${lx}" y="5" width="9" height="9" rx="2" fill="${s.color}"/><text class="legend-text" x="${lx + 14}" y="13">${esc(s.name)}</text>`;
      lx += 22 + s.name.length * 6.4;
    });
    svg += `<line class="cross" x1="0" x2="0" y1="${pt}" y2="${H - pb}" stroke="rgba(56,189,248,.45)" stroke-dasharray="3,3" style="display:none"/>`;
    series.forEach((s, i) => { svg += `<circle class="hl hl-${i}" r="5" fill="${s.color}" stroke="#fff" stroke-width="1.5" style="display:none"/>`; });
    svg += '</svg><div class="chart-tip" hidden></div>';
    host.innerHTML = svg;

    const el = $('svg', host);
    const tip = $('.chart-tip', host);
    const cross = $('.cross', host);
    let cur = -1;
    function hide() {
      cur = -1; tip.hidden = true; cross.style.display = 'none';
      series.forEach((_, i) => { $('.hl-' + i, host).style.display = 'none'; });
    }
    function showAt(i) {
      i = Math.max(0, Math.min(xs.length - 1, i));
      cur = i;
      const x = xs[i];
      cross.setAttribute('x1', sx(x)); cross.setAttribute('x2', sx(x)); cross.style.display = '';
      let rows = '';
      let anchorY = H / 2;
      series.forEach((s, si) => {
        const p = s.points.find((q) => q.x === x);
        const hl = $('.hl-' + si, host);
        if (p) {
          hl.setAttribute('cx', sx(x)); hl.setAttribute('cy', sy(p.y)); hl.style.display = '';
          rows += `<div><i style="background:${s.color}"></i>${esc(s.name)}: <b>${esc(yFmt(p.y, true))}</b></div>`;
          anchorY = sy(p.y);
        } else { hl.style.display = 'none'; }
      });
      tip.innerHTML = `<div><b>EPOCH ${x}</b></div>${rows}`;
      tip.hidden = false;
      const m = el.getScreenCTM();
      const pt0 = el.createSVGPoint(); pt0.x = sx(x); pt0.y = anchorY;
      const c = pt0.matrixTransform(m);
      const hr = host.getBoundingClientRect();
      let left = c.x - hr.left + 12;
      if (left + tip.offsetWidth > hr.width - 4) left = c.x - hr.left - tip.offsetWidth - 12;
      let top = c.y - hr.top - tip.offsetHeight / 2;
      top = Math.max(0, Math.min(top, hr.height - tip.offsetHeight));
      tip.style.left = Math.max(0, left) + 'px';
      tip.style.top = top + 'px';
    }
    el.addEventListener('pointermove', (e) => {
      const pt1 = el.createSVGPoint(); pt1.x = e.clientX; pt1.y = e.clientY;
      const p = pt1.matrixTransform(el.getScreenCTM().inverse());
      let best = 0; let bd = Infinity;
      xs.forEach((x, i) => { const d = Math.abs(sx(x) - p.x); if (d < bd) { bd = d; best = i; } });
      showAt(best);
    });
    el.addEventListener('pointerleave', hide);
    el.addEventListener('focus', () => showAt(xs.length - 1));
    el.addEventListener('blur', hide);
    el.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight') { e.preventDefault(); showAt(cur < 0 ? 0 : cur + 1); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); showAt(cur < 0 ? xs.length - 1 : cur - 1); }
    });
  }

  /* =====================================================================
   * 1. DEMO: Top telemetry, three-pane gauge, recorder, verify
   * ===================================================================== */
  const COLORS = { s0: '#22d3ee', s1: '#38bdf8', warn: '#fbbf24', same: '#34d399', diff: '#fb7185' };
  const verifyBtn = $('#verify-btn');
  const micAlert = $('#mic-alert');
  const errorAlert = $('#error-alert');
  const shortWarnAlert = $('#short-warn');
  const MAX_RECORD_MS = 10000;
  let enrollBlob = null, testBlob = null, enrollName = 'enrollment.webm', testName = 'test.webm';
  let enrollDur = null, testDur = null;

  function showMic(msg) { micAlert.textContent = msg; micAlert.className = 'alert warn'; }
  function clearError() { errorAlert.textContent = ''; errorAlert.className = 'alert'; }
  function showError(msg) { errorAlert.textContent = msg; errorAlert.className = 'alert error'; }
  function updateVerifyState() { verifyBtn.disabled = !(enrollBlob && testBlob); }

  function updateShortWarn() {
    if (!shortWarnAlert) return;
    const isShortEnroll = enrollDur != null && isFinite(enrollDur) && enrollDur >= 1.0 && enrollDur < 3.0;
    const isShortTest = testDur != null && isFinite(testDur) && testDur >= 1.0 && testDur < 3.0;
    if (isShortEnroll || isShortTest) {
      shortWarnAlert.textContent = 'Short recording: Clips under 3 seconds may not contain enough speech information for reliable speaker verification. Try recording a little longer.';
      shortWarnAlert.className = 'alert warn';
      shortWarnAlert.hidden = false;
    } else {
      shortWarnAlert.textContent = '';
      shortWarnAlert.className = 'alert';
      shortWarnAlert.hidden = true;
    }
  }

  /* ---- gauge ---- */
  const G = { cx: 150, cy: 140, r: 108 };
  let gaugeThr = null;
  const polar = (v, r) => {
    const a = Math.PI * (1 - (Math.max(-1, Math.min(1, v)) + 1) / 2);
    return [G.cx + (r || G.r) * Math.cos(a), G.cy - (r || G.r) * Math.sin(a)];
  };
  const arcPath = (v1, v2) => {
    const a = polar(v1), b = polar(v2);
    return `M${a[0].toFixed(2)} ${a[1].toFixed(2)} A${G.r} ${G.r} 0 0 1 ${b[0].toFixed(2)} ${b[1].toFixed(2)}`;
  };
  function buildGauge(thr, margin) {
    gaugeThr = thr;
    let s = '<svg viewBox="0 0 300 184" role="img" aria-label="Score gauge from -1 to 1 with the decision threshold and borderline zone">';
    s += `<circle cx="150" cy="140" r="88" fill="none" stroke="rgba(56,189,248,0.1)" stroke-dasharray="2,4"/>`;
    if (thr == null) {
      s += `<path d="${arcPath(-1, 1)}" fill="none" stroke="rgba(255,255,255,.14)" stroke-width="14"/>`;
    } else {
      const lo = Math.max(-1, thr - margin), hi = Math.min(1, thr + margin);
      s += `<path d="${arcPath(-1, lo)}" fill="none" stroke="${COLORS.diff}" stroke-opacity=".78" stroke-width="14"/>`;
      s += `<path d="${arcPath(lo, hi)}" fill="none" stroke="${COLORS.warn}" stroke-width="14"/>`;
      s += `<path d="${arcPath(hi, 1)}" fill="none" stroke="${COLORS.same}" stroke-opacity=".82" stroke-width="14"/>`;
      const t1 = polar(thr, G.r - 15), t2 = polar(thr, G.r + 15), tl = polar(thr, G.r + 27);
      s += `<line x1="${t1[0]}" y1="${t1[1]}" x2="${t2[0]}" y2="${t2[1]}" stroke="#22d3ee" stroke-width="2.5"/>`;
      s += `<text x="${tl[0]}" y="${tl[1]}" fill="#22d3ee" font-family="ui-monospace, monospace" font-size="10" font-weight="700" text-anchor="middle">${thr.toFixed(2)}</text>`;
    }
    [[-1, '-1.0'], [0, '0.0'], [1, '+1.0']].forEach(([v, l]) => {
      const p = polar(v, G.r - 24);
      s += `<text x="${p[0]}" y="${p[1] + 3}" fill="#97aacb" font-family="ui-monospace, monospace" font-size="9.5" text-anchor="middle">${l}</text>`;
    });
    s += `<g id="needle"><line x1="150" y1="140" x2="150" y2="48" stroke="#22d3ee" stroke-width="3.2" stroke-linecap="round" style="filter:drop-shadow(0 0 6px rgba(34,211,238,.95))"/><circle cx="150" cy="140" r="7" fill="#22d3ee" stroke="#031922" stroke-width="2"/></g>`;
    s += '<text id="g-score" x="150" y="174" text-anchor="middle" fill="#22d3ee" font-family="ui-monospace, monospace" font-size="22" font-weight="800">—</text>';
    s += '</svg>';
    $('#gauge-wrap').innerHTML = s;
    $('#gauge-legend').innerHTML =
      `<span><i style="background:${COLORS.diff}"></i>DIFFERENT</span>` +
      `<span><i style="background:${COLORS.warn}"></i>BORDERLINE</span>` +
      `<span><i style="background:${COLORS.same}"></i>SAME</span>`;
  }
  function animateScore(target) {
    const el = $('#g-score');
    if (!el) return;
    if (reduceMotion()) { el.textContent = target.toFixed(4); return; }
    const t0 = performance.now(), dur = 900, from = -1;
    (function tick(now) {
      const k = Math.min(1, (now - t0) / dur);
      const e = 1 - Math.pow(1 - k, 3);
      el.textContent = (from + (target - from) * e).toFixed(4);
      if (k < 1) requestAnimationFrame(tick); else el.textContent = target.toFixed(4);
    })(t0);
  }
  function setVerdict(text, state, reveal) {
    const v = $('#verdict');
    v.textContent = text;
    v.className = 'verdict ' + state;
    if (reveal && !reduceMotion()) { void v.offsetWidth; v.classList.add('reveal'); }
  }

  function showResult(d) {
    const thr = Number(d.threshold), score = Number(d.score);
    const margin = Number(!isNil(d.borderline_margin) ? d.borderline_margin : 0.10);
    const width = Number(!isNil(d.borderline_width) ? d.borderline_width : 2 * margin);
    const band = d.borderline_band || { low: thr - margin, high: thr + margin };
    if (gaugeThr !== thr) buildGauge(thr, margin);
    $('#res-score').textContent = score.toFixed(4);
    $('#res-threshold').textContent = thr.toFixed(4);
    const zone = d.borderline_zone || `threshold +/- ${margin.toFixed(2)}`;
    $('#res-borderline').textContent = `${zone} (width ${width.toFixed(2)}: ${Number(band.low).toFixed(4)} to ${Number(band.high).toFixed(4)})`;
    $('#res-enroll-dur').textContent = num(d.enrollment_duration, 2) + ' s';
    $('#res-test-dur').textContent = num(d.test_duration, 2) + ' s';
    const ckptPath = (DATA.info && DATA.info.checkpoint_path) || '';
    const ckptEp = !isNil(d.checkpoint_epoch) ? d.checkpoint_epoch : (DATA.info && DATA.info.checkpoint_epoch);
    if (ckptPath || !isNil(ckptEp)) {
      $('#res-ckpt-epoch').textContent = `${ckptPath ? ckptPath + ' · ' : ''}ep ${ckptEp}`;
    }
    const devEer = !isNil(d.dev_eer) ? d.dev_eer : (DATA.info && DATA.info.dev_eer);
    if (!isNil(devEer)) {
      $('#res-dev-eer').textContent = num(devEer, 2) + '%';
    }
    if (!isNil(d.enrollment_duration)) enrollDur = Number(d.enrollment_duration);
    if (!isNil(d.test_duration)) testDur = Number(d.test_duration);
    updateShortWarn();
    const needle = $('#needle');
    setVerdict('…', 'idle', false);
    requestAnimationFrame(() => { needle.style.transform = `rotate(${Math.max(-1, Math.min(1, score)) * 90}deg)`; });
    animateScore(score);
    const state = String(d.verdict || '').toLowerCase();
    setTimeout(() => setVerdict(d.verdict, state, true), reduceMotion() ? 0 : 750);
  }

  /* ---- verification request ---- */
  async function runVerification() {
    clearError();
    if (!enrollBlob || !testBlob) { showError('Please record or choose both an enrollment clip and a test clip first.'); return; }
    verifyBtn.disabled = true;
    const label = verifyBtn.textContent;
    verifyBtn.textContent = 'Analyzing…';
    $('#analyzing').hidden = false;
    $('#result-card').setAttribute('aria-busy', 'true');
    try {
      const fd = new FormData();
      fd.append('enrollment', enrollBlob, enrollName);
      fd.append('test', testBlob, testName);
      const [resp] = await Promise.all([fetch('/api/verify', { method: 'POST', body: fd }), sleep(450)]);
      const data = await resp.json();
      if (!resp.ok) { showError(data.error || data.detail || 'Verification failed.'); return; }
      showResult(data);
    } catch (err) {
      showError('Could not reach the local server: ' + err.message);
    } finally {
      $('#analyzing').hidden = true;
      $('#result-card').removeAttribute('aria-busy');
      verifyBtn.textContent = label;
      updateVerifyState();
    }
  }

  function setSlot(slot, blob, filename, measuredDur) {
    const url = URL.createObjectURL(blob);
    const kb = (blob.size / 1024).toFixed(1);
    if (slot === 'enrollment') {
      enrollBlob = blob; enrollName = filename;
      enrollDur = (measuredDur != null && isFinite(measuredDur)) ? Number(measuredDur) : null;
      const audioEl = $('#enroll-audio');
      audioEl.src = url;
      audioEl.hidden = false;
      $('#enroll-status').textContent = `Loaded: ${filename} (${kb} KB)`;
    } else {
      testBlob = blob; testName = filename;
      testDur = (measuredDur != null && isFinite(measuredDur)) ? Number(measuredDur) : null;
      const audioEl = $('#test-audio');
      audioEl.src = url;
      audioEl.hidden = false;
      $('#test-status').textContent = `Loaded: ${filename} (${kb} KB)`;
    }
    updateShortWarn();
    updateVerifyState();
  }

  /* ---- microphone recorder with live level meter ---- */
  function setupRecorder(slot, btnId, timerId, levelId, meterId, fileId) {
    const btn = $('#' + btnId), timerEl = $('#' + timerId), levelEl = $('#' + levelId), meterEl = $('#' + meterId);
    const labelEl = $('.rec-label', btn);
    let rec = null, chunks = [], ctx = null, raf = null, stopT = null, t0 = 0;
    const idleLabel = 'Record (up to 10 s)';

    function cleanup(stream) {
      if (raf) { cancelAnimationFrame(raf); raf = null; }
      if (stopT) { clearTimeout(stopT); stopT = null; }
      if (stream) stream.getTracks().forEach((t) => t.stop());
      if (ctx && ctx.state !== 'closed') ctx.close().catch(() => {});
      meterEl.style.width = '0%'; levelEl.textContent = 'Level 0%';
    }
    btn.addEventListener('click', async () => {
      clearError();
      if (rec && rec.state === 'recording') { rec.stop(); return; }
      if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== 'function' || typeof window.MediaRecorder === 'undefined') {
        showMic('This browser cannot record from a microphone here. Please use "Or upload an audio file" below, or the ready-made example buttons.');
        return;
      }
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      } catch (err) {
        const n = err && err.name;
        if (n === 'NotAllowedError' || n === 'PermissionDeniedError' || n === 'SecurityError') {
          showMic('Microphone access was blocked. You can allow it from the lock icon in the address bar and try again, or simply use "Or upload an audio file" below.');
        } else if (n === 'NotFoundError' || n === 'DevicesNotFoundError' || n === 'OverconstrainedError') {
          showMic('No microphone was found. Please connect one and try again, or use "Or upload an audio file" below.');
        } else {
          showMic('The microphone could not be started (' + ((err && (err.message || err.name)) || 'unknown error') + '). Please use "Or upload an audio file" below.');
        }
        return;
      }
      micAlert.className = 'alert'; micAlert.textContent = '';
      try { chunks = []; rec = new MediaRecorder(stream); }
      catch (err) { cleanup(stream); showMic('Recording could not be initialised (' + err.message + '). Please use "Or upload an audio file" below.'); return; }
      try {
        const AC = window.AudioContext || window.webkitAudioContext;
        ctx = new AC();
        const an = ctx.createAnalyser(); an.fftSize = 512;
        ctx.createMediaStreamSource(stream).connect(an);
        const buf = new Uint8Array(an.fftSize);
        t0 = performance.now();
        (function loop() {
          if (!rec || rec.state !== 'recording') return;
          an.getByteTimeDomainData(buf);
          let sum = 0;
          for (let i = 0; i < buf.length; i++) { const v = (buf[i] - 128) / 128; sum += v * v; }
          const pct = Math.min(100, Math.round(Math.sqrt(sum / buf.length) * 300));
          meterEl.style.width = pct + '%'; levelEl.textContent = `Level ${pct}%`;
          timerEl.textContent = `Recording ${Math.min(10, (performance.now() - t0) / 1000).toFixed(1)} s / 10 s`;
          raf = requestAnimationFrame(loop);
        })();
      } catch (_e) { /* meter is optional */ }
      rec.ondataavailable = (ev) => { if (ev.data && ev.data.size > 0) chunks.push(ev.data); };
      rec.onstop = () => {
        const mime = rec.mimeType || 'audio/webm';
        const ext = mime.indexOf('ogg') >= 0 ? 'ogg' : mime.indexOf('mp4') >= 0 ? 'mp4' : 'webm';
        const blob = new Blob(chunks, { type: mime });
        const durSec = (performance.now() - t0) / 1000;
        cleanup(stream);
        labelEl.textContent = idleLabel; btn.classList.remove('recording');
        timerEl.textContent = 'Recorded clip ready';
        setSlot(slot, blob, `${slot}_recording.${ext}`, durSec);
      };
      rec.start(100);
      labelEl.textContent = 'Stop recording'; btn.classList.add('recording');
      stopT = setTimeout(() => { if (rec && rec.state === 'recording') rec.stop(); }, MAX_RECORD_MS);
    });
    $('#' + fileId).addEventListener('change', (e) => {
      clearError();
      const f = e.target.files && e.target.files[0];
      if (f) setSlot(slot, f, f.name);
    });
  }

  async function loadExample(id, nameA, nameB) {
    clearError();
    try {
      const [a, b] = await Promise.all([fetch(`/api/examples/${id}/enrollment`), fetch(`/api/examples/${id}/test`)]);
      if (!a.ok || !b.ok) throw new Error('The local LibriSpeech example files could not be loaded.');
      const [ba, bb] = await Promise.all([a.blob(), b.blob()]);
      setSlot('enrollment', ba, nameA); setSlot('test', bb, nameB);
      await runVerification();
    } catch (err) { showError(err.message); }
  }

  setupRecorder('enrollment', 'enroll-rec-btn', 'enroll-timer', 'enroll-level-text', 'enroll-meter', 'enroll-file');
  setupRecorder('test', 'test-rec-btn', 'test-timer', 'test-level-text', 'test-meter', 'test-file');
  $('#enroll-audio').addEventListener('loadedmetadata', (e) => {
    const d = e.currentTarget.duration;
    if (isFinite(d) && d > 0) {
      enrollDur = d;
      updateShortWarn();
    }
  });
  $('#test-audio').addEventListener('loadedmetadata', (e) => {
    const d = e.currentTarget.duration;
    if (isFinite(d) && d > 0) {
      testDur = d;
      updateShortWarn();
    }
  });
  verifyBtn.addEventListener('click', runVerification);
  $('#example-same-btn').addEventListener('click', () => loadExample('same', '7176-88083-0008.flac', '7176-92135-0037.flac'));
  $('#example-diff-btn').addEventListener('click', () => loadExample('different', '4970-29095-0030.flac', '4992-41797-0000.flac'));
  if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== 'function' || typeof window.MediaRecorder === 'undefined') {
    showMic('Microphone recording is not available in this browser or context (use http://127.0.0.1 or http://localhost in a current browser). You can still upload audio files or use the ready-made example buttons.');
  }
  buildGauge(null, 0.10);

  function renderDemoFacts() {
    const info = DATA.info;
    if (info) {
      const thr = Number(info.threshold);
      const margin = Number(!isNil(info.borderline_margin) ? info.borderline_margin : 0.10);
      const width = Number(!isNil(info.borderline_width) ? info.borderline_width : 2 * margin);
      const lo = !isNil(info.borderline_low) ? Number(info.borderline_low) : thr - margin;
      const hi = !isNil(info.borderline_high) ? Number(info.borderline_high) : thr + margin;
      const paramStr = !isNil(info.parameter_count) ? ` (${fmt(info.parameter_count)} params)` : '';
      const labelPrefix = info.model_label || "Scaled reproduction on LibriSpeech train-clean-100, not the paper's model";
      $('#model-line').innerHTML = `<strong>${esc(labelPrefix)}:</strong> ${esc(info.checkpoint_path)}${paramStr} · epoch ${esc(info.checkpoint_epoch)} · dev-clean ${term('EER')} ${num(info.dev_eer, 2)}% ${tagHtml('measured')}`;
      if ($('#nav-val-model')) $('#nav-val-model').textContent = info.model_name || 'ska_hnn (S0)';
      $('#nav-val-ckpt').textContent = `ep ${info.checkpoint_epoch} · ${num(info.dev_eer, 2)}% EER`;
      $('#res-threshold').textContent = thr.toFixed(4);
      $('#res-borderline').textContent = `${info.borderline_zone || `threshold +/- ${margin.toFixed(2)}`} (width ${width.toFixed(2)}: ${lo.toFixed(4)} to ${hi.toFixed(4)})`;
      $('#res-ckpt-epoch').textContent = `${info.checkpoint_path} · ep ${info.checkpoint_epoch}`;
      $('#res-dev-eer').textContent = `${num(info.dev_eer, 2)}%`;
      buildGauge(thr, margin);
    } else {
      $('#model-line').textContent = "Scaled reproduction on LibriSpeech train-clean-100, not the paper's model: unavailable (could not read /api/info).";
    }
    const meta = DATA.results && DATA.results.runs && DATA.results.runs.s0 && DATA.results.runs.s0.meta;
    if (meta && !isNil(meta.num_speakers)) {
      const chip = $('#data-chip');
      chip.textContent = `LibriSpeech train-clean-100 · ${fmt(meta.num_speakers)} speakers`;
      chip.hidden = false;
      $('#nav-val-dataset').textContent = `LibriSpeech (${fmt(meta.num_speakers)} spk)`;
    }
  }

  /* =====================================================================
   * 2. HOW IT WORKS: Compact 5-stage flowchart + hand-drawn SVG icons + strip
   * ===================================================================== */
  const STAGE_SVGS = {
    data: `<svg class="stage-icon" viewBox="0 0 56 32" aria-hidden="true"><path d="M2 16h5l3-8 4 16 4-20 4 24 4-14 3 8 4-11 4 14 3-6h14" fill="none" stroke="#22d3ee" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    logmel: `<svg class="stage-icon" viewBox="0 0 56 32" aria-hidden="true"><g fill="#38bdf8" opacity="0.88"><rect x="3" y="18" width="5" height="11" rx="1"/><rect x="11" y="10" width="5" height="19" rx="1"/><rect x="19" y="5" width="5" height="24" rx="1" fill="#22d3ee"/><rect x="27" y="12" width="5" height="17" rx="1"/><rect x="35" y="7" width="5" height="22" rx="1" fill="#34d399"/><rect x="43" y="15" width="5" height="14" rx="1"/></g></svg>`,
    backbone: `<svg class="stage-icon" viewBox="0 0 56 32" aria-hidden="true"><g stroke="#22d3ee" stroke-width="1.4" fill="rgba(34,211,238,0.16)"><rect x="3" y="6" width="11" height="20" rx="2"/><rect x="22" y="4" width="12" height="24" rx="2"/><rect x="42" y="8" width="11" height="16" rx="2"/><line x1="14" y1="12" x2="22" y2="10"/><line x1="14" y1="20" x2="22" y2="22"/><line x1="34" y1="12" x2="42" y2="13"/><line x1="34" y1="20" x2="42" y2="19"/></g></svg>`,
    embedding: `<svg class="stage-icon" viewBox="0 0 56 32" aria-hidden="true"><circle cx="18" cy="16" r="11" fill="rgba(34,211,238,0.1)" stroke="#38bdf8" stroke-width="1.4" stroke-dasharray="3,2"/><line x1="18" y1="16" x2="27" y2="9" stroke="#34d399" stroke-width="2" stroke-linecap="round"/><circle cx="27" cy="9" r="2.3" fill="#34d399"/><g fill="#22d3ee"><rect x="35" y="7" width="17" height="3.5" rx="1"/><rect x="35" y="14" width="13" height="3.5" rx="1"/><rect x="35" y="21" width="19" height="3.5" rx="1"/></g></svg>`,
    loss: `<svg class="stage-icon" viewBox="0 0 56 32" aria-hidden="true"><path d="M8 27 A20 20 0 0 1 48 27" fill="none" stroke="rgba(56,189,248,0.35)" stroke-width="1.5"/><line x1="28" y1="27" x2="17" y2="10" stroke="#34d399" stroke-width="1.8"/><line x1="28" y1="27" x2="39" y2="10" stroke="#22d3ee" stroke-width="1.8"/><path d="M21 16 A12 12 0 0 1 35 16" fill="none" stroke="#fbbf24" stroke-width="1.5" stroke-dasharray="2,2"/><circle cx="45" cy="9" r="2.2" fill="#fb7185"/></svg>`,
  };

  const arch = () => (DATA.results && DATA.results.architecture) || {};
  const contracts = (prefix) => (arch().stage_contracts || []).filter((c) => String(c.stage).indexOf(prefix) === 0);
  const moduleById = (id) => (arch().clickable_modules || []).find((m) => m.id === id);
  const paramCell = (v) => (isNil(v) ? '—' : fmt(v));

  function contractTable(rows) {
    if (!rows.length) return '';
    let h = '<div class="table-wrap"><table><thead><tr><th>Module</th><th>Goes in</th><th>Comes out</th><th class="n">Parameters</th></tr></thead><tbody>';
    rows.forEach((c) => {
      h += `<tr><td>${rich(c.module_name, true)}</td><td><code>${esc(clean(c.input_shape))}</code></td><td><code>${esc(clean(c.output_shape))}</code></td>` +
        `<td class="n">${paramCell(c.scaled_params)}</td></tr>`;
    });
    return h + '</tbody></table></div><p class="muted small">Shapes: B = batch, T = time frames. Parameter counts are measured with torchinfo ' + tagHtml('measured') + '.</p>';
  }

  function stageDefs() {
    const meta = (DATA.results.runs.s0 && DATA.results.runs.s0.meta) || {};
    const fc1 = contracts('5.').find((c) => /FC Layer 1/i.test(c.module_name));
    const iaam = contracts('5.').find((c) => /IAAMSoftmax/i.test(c.module_name));
    const embMatch = fc1 && /(\d+)\)?\s*$/.exec(clean(fc1.output_shape));
    const embDim = embMatch ? embMatch[1] : null;
    const fe = contracts('0.')[0];
    const aug = meta.augmentation_flags ? Object.keys(meta.augmentation_flags).filter((k) => meta.augmentation_flags[k]) : [];
    const augOff = meta.augmentation_flags ? Object.keys(meta.augmentation_flags).filter((k) => !meta.augmentation_flags[k]) : [];
    const tier3 = (DATA.results.tiers || [])[3];

    return [
      {
        id: 'data', title: 'Speech audio',
        io: `<b>In:</b> a recording of someone talking. <b>Out:</b> short snippets of raw audio samples for the model.`,
        foot: !isNil(meta.dataset_size) ? `${fmt(meta.dataset_size)} utterances · ${fmt(meta.num_speakers)} speakers ${tagHtml('measured')}` : '',
        body: () => `<p>The model is trained on LibriSpeech <code>train-clean-100</code>${!isNil(meta.dataset_size) ? `: ${fmt(meta.dataset_size)} utterances from ${fmt(meta.num_speakers)} speakers ${tagHtml('measured')}` : ''}. Each training step draws a batch of ${fmt(meta.batch_size)} short clips from different speakers ${tagHtml('measured')}.</p>` +
          (aug.length ? `<p>Augmentations switched on during training: <b>${esc(aug.join(', '))}</b>${augOff.length ? `; switched off: ${esc(augOff.join(', '))}` : ''} ${tagHtml('measured')}.</p>` : '') +
          '<p>At evaluation time each clip is used at its <b>full length</b>, with no cropping.</p>',
      },
      {
        id: 'logmel', title: 'Log-Mel spectrogram',
        io: `<b>In:</b> the raw audio. <b>Out:</b> an 80-band picture of the sound over time (a ${term('log-Mel', 'log-Mel spectrogram')}).`,
        foot: fe ? `Learnable parameters: ${paramCell(fe.scaled_params)} ${tagHtml('measured')}` : '',
        body: () => `<p>The audio is converted to a ${term('log-Mel')} spectrogram: energy in 80 frequency bands for every 10 ms of sound. Training adds random masking (SpecAugment) so the model does not rely on any single band. This step has no learnable parameters.</p>` + contractTable(contracts('0.')),
      },
      {
        id: 'backbone', title: 'SKA-HNN backbone',
        io: `<b>In:</b> the spectrogram. <b>Out:</b> one compact summary of the voice for the whole clip.`,
        foot: !isNil(meta.parameter_count) ? `Whole model: ${fmt(meta.parameter_count)} ${term('parameters')} ${tagHtml('measured')}` : '',
        body: () => {
          const ids = ['fwSKA', 'cwSKA', 'SE', 'ResBlock', 'local_stats', 'TDNN-LSTM', 'Fully-TDNN', 'MLFC', 'MLASP'];
          const labels = { local_stats: 'Local statistics' };
          let mods = '<h3>Building blocks (click to expand)</h3>';
          ids.forEach((id) => {
            const m = moduleById(id);
            if (!m) return;
            mods += `<details class="mod"><summary><span>${esc(labels[id] || m.id)}</span><span class="mono">${fmt(m.params_total)} ${tagHtml('measured')}</span></summary>` +
              `<p>${rich(m.description, true)}</p><p class="small">Shapes: <code>${esc(clean(m.input_shape))}</code> → <code>${esc(clean(m.output_shape))}</code>; one instance has ${fmt(m.params_single)} parameters.</p></details>`;
          });
          return `<p>A 2-D convolutional front end (with ${term('SKA')} attention blocks) feeds a ${term('TDNN')}-LSTM stage and a fully-TDNN stage; their outputs are combined (${term('MLFC')}) and pooled over time (${term('MLASP')}) into one summary vector.</p>` +
            contractTable(['1.', '2.', '3.', '4.'].flatMap((p) => contracts(p))) + mods;
        },
      },
      {
        id: 'embedding', title: 'Speaker embedding',
        io: `<b>In:</b> that summary. <b>Out:</b> ${embDim ? `a ${esc(embDim)}-number` : 'a fixed-length'} voiceprint (the ${term('embedding')}) used to compare speakers.`,
        foot: fc1 ? `First linear layer: ${paramCell(fc1.scaled_params)} ${tagHtml('measured')}` : '',
        body: () => `<p>A linear layer squeezes the pooled summary into the ${term('embedding')}. At comparison time the two embeddings are length-normalised and compared with the ${term('cosine score')}; the demo then applies the ${term('threshold')} and the ${term('borderline zone')}.</p>` + contractTable(contracts('5.').filter((c) => /FC Layer/i.test(c.module_name))),
      },
      {
        id: 'loss', title: 'IAAMSoftmax with AdaptiveDrop',
        io: `<b>In:</b> embeddings and the true speaker labels (training only). <b>Out:</b> a learning signal that separates speakers while ignoring suspicious clips.`,
        foot: iaam ? `Speaker prototypes: ${paramCell(iaam.scaled_params)} ${tagHtml('measured')}` : '',
        body: () => `<p>${term('IAAMSoftmax')} is the training loss. ${term('AdaptiveDrop')} watches how each clip scores against its labelled speaker and, after a warm-up, filters out clips that look mislabeled or relabels them. It is used for training only; the demo does not use it.</p>` +
          (tier3 ? `<p class="small">${rich(tier3.details, true)}</p>` : '') + contractTable(contracts('5.').filter((c) => /IAAMSoftmax/i.test(c.module_name))),
      },
    ];
  }

  function renderPipeline() {
    const defs = stageDefs();
    const host = $('#pipeline');
    host.innerHTML = '';
    defs.forEach((d, i) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'stage-card corner-card';
      b.setAttribute('aria-haspopup', 'dialog');
      b.innerHTML =
        `<div class="stage-top"><span class="stage-num">STAGE 0${i + 1}</span>${STAGE_SVGS[d.id] || ''}</div>` +
        `<h3>${esc(d.title)}</h3>` +
        `<p class="stage-io">${d.io}</p>` +
        `<div class="stage-foot"><span>${d.foot}</span><span class="stage-more">Details →</span></div>`;
      b.addEventListener('click', () => openModal(`Stage ${i + 1}: ${d.title}`, `<p>${d.io}</p>${d.body()}`, b, `STAGE 0${i + 1} TELEMETRY`));
      host.appendChild(b);
      if (i < defs.length - 1) {
        const a = document.createElement('div');
        a.className = 'pipe-arrow';
        a.setAttribute('aria-hidden', 'true');
        a.innerHTML = `<svg viewBox="0 0 26 18"><line class="pipe-dash" x1="1" y1="9" x2="20" y2="9" stroke="#22d3ee" stroke-width="1.8"/><polygon points="18,4 25,9 18,14" fill="#22d3ee"/></svg>`;
        host.appendChild(a);
      }
    });
    $$('.stage-card .term', host).forEach((t) => { t.removeAttribute('tabindex'); });

    // Slim "one clip in, one decision out" strip below the row
    const meta = (DATA.results.runs.s0 && DATA.results.runs.s0.meta) || {};
    const fc1 = contracts('5.').find((c) => /FC Layer 1/i.test(c.module_name));
    const iaam = contracts('5.').find((c) => /IAAMSoftmax/i.test(c.module_name));
    const strip = $('#flow-strip');
    if (strip) {
      strip.innerHTML =
        `<span class="flow-strip-tag">ONE CLIP IN, ONE DECISION OUT</span>` +
        `<div class="flow-strip-nodes">` +
        `<span class="flow-node"><b>1.</b> Speech audio (${fmt(meta.dataset_size)} utts · ${fmt(meta.num_speakers)} spk)</span>` +
        `<span class="flow-sep">→</span>` +
        `<span class="flow-node"><b>2.</b> 80-band log-Mel (0 params)</span>` +
        `<span class="flow-sep">→</span>` +
        `<span class="flow-node"><b>3.</b> SKA-HNN (${fmt(meta.parameter_count)} params)</span>` +
        `<span class="flow-sep">→</span>` +
        `<span class="flow-node"><b>4.</b> 192-d embedding (${fc1 ? fmt(fc1.scaled_params) : '—'} params)</span>` +
        `<span class="flow-sep">→</span>` +
        `<span class="flow-node"><b>5.</b> Cosine score vs threshold · IAAMSoftmax (${iaam ? fmt(iaam.scaled_params) : '—'} params)</span>` +
        `</div>` +
        tagHtml('measured');
    }
  }

  /* =====================================================================
   * 3. RESULTS (No S1 pill; Training time modal shows ONLY measured Mac rows)
   * ===================================================================== */
  function epochTable(run, title) {
    const rows = runEpochs(run);
    const ad = rows.some((r) => !isNil(r.loss_keep_ratio));
    let h = `<h3>${esc(title)}</h3><div class="table-wrap"><table><thead><tr><th class="n">Epoch</th><th class="n">Train loss</th><th class="n">Train acc %</th><th class="n">Dev-clean EER %</th><th class="n">Dev minDCF (P=0.05)</th><th class="n">LR</th>` +
      (ad ? '<th class="n">Keep ratio</th><th class="n">Filtered</th><th class="n">Corrected</th>' : '') + '</tr></thead><tbody>';
    rows.forEach((r) => {
      h += `<tr><td class="n">${r.epoch}</td><td class="n">${num(r.train_loss, 4)}</td><td class="n">${num(r.train_acc, 2)}</td>` +
        `<td class="n">${num(r.dev_eer_pct, 2)}</td><td class="n">${num(r.dev_mindcf_05, 4)}</td><td class="n">${isNil(r.lr) ? '—' : Number(r.lr).toExponential(1)}</td>` +
        (ad ? `<td class="n">${isNil(r.loss_keep_ratio) ? '—' : num(Number(r.loss_keep_ratio) * 100, 2) + '%'}</td><td class="n">${fmt(r.loss_num_filtered)}</td><td class="n">${fmt(r.loss_num_corrected)}</td>` : '') + '</tr>';
    });
    return h + '</tbody></table></div>';
  }

  function openViewData(opener) {
    const runs = DATA.results.runs;
    let h = `<p class="muted small">Every epoch logged in <code>history.csv</code>. Dev-clean columns are filled only on epochs where an evaluation was run. From the epoch where ${term('AdaptiveDrop')} starts, S0's train loss is computed on the retained clips only, so it is not comparable with S1's.</p>`;
    if (runs.s0) {
      const p = progress(runs.s0);
      h += epochTable(runs.s0, `S0 (IAAMSoftmax + AdaptiveDrop) — ${p.k}/${p.total} epochs logged`);
    }
    if (runs.s1) {
      const p = progress(runs.s1);
      h += epochTable(runs.s1, `S1 (AAMSoftmax) — ${p.done ? 'trained' : 'in progress'}: ${p.k}/${p.total} epochs logged`);
    }
    openModal('Per-epoch data', h, opener, 'EPOCH LEDGER · HISTORY.CSV');
  }

  /* Training time modal: show ONLY measured Mac rows; drop every Colab / not-run-yet / pending row. */
  function openTiming(opener) {
    const runs = DATA.results.runs, b = DATA.results.benchmarks || {};
    const rows = [];
    [['Mac M3 Pro · S0 pilot (1 epoch)', runs.pilot_s0_mac], ['Mac M3 Pro · S1 pilot (1 epoch)', runs.pilot_s1_mac]].forEach(([label, run]) => {
      const e = run && run.epochs && run.epochs[0];
      if (e) rows.push({ label, dw: e.data_loading_sec, comp: e.forward_backward_optimizer_sec, ev: e.dev_clean_eval_sec, total: e.total_sec });
    });
    const maxT = Math.max.apply(null, rows.map((r) => Number(r.total) || 0)) || 1;
    const lh = 38, W = 820, left = 240, bw = 480;
    let svg = `<svg viewBox="0 0 ${W} ${rows.length * lh + 34}" style="width:100%;height:auto" role="img" aria-label="Seconds per epoch by Mac M3 Pro pilot run">`;
    svg += `<rect x="${left}" y="6" width="10" height="10" fill="#94a3b8"/><text x="${left + 15}" y="15" font-family="ui-monospace, monospace" font-size="11" fill="#e8f0fe">Data wait</text>`;
    svg += `<rect x="${left + 95}" y="6" width="10" height="10" fill="#22d3ee"/><text x="${left + 110}" y="15" font-family="ui-monospace, monospace" font-size="11" fill="#e8f0fe">Compute</text>`;
    svg += `<rect x="${left + 185}" y="6" width="10" height="10" fill="#34d399"/><text x="${left + 200}" y="15" font-family="ui-monospace, monospace" font-size="11" fill="#e8f0fe">Dev-clean eval</text>`;
    rows.forEach((r, i) => {
      const y = 30 + i * lh;
      svg += `<text x="${left - 10}" y="${y + 15}" text-anchor="end" font-family="ui-monospace, monospace" font-size="11" fill="#97aacb">${esc(r.label)}</text>`;
      const w1 = Math.max(3, (r.dw / maxT) * bw), w2 = (r.comp / maxT) * bw, w3 = (r.ev / maxT) * bw;
      svg += `<rect x="${left}" y="${y}" width="${w1}" height="22" fill="#94a3b8"><title>Data wait ${num(r.dw)} s</title></rect>`;
      svg += `<rect x="${left + w1}" y="${y}" width="${w2}" height="22" fill="#22d3ee"><title>Compute ${num(r.comp)} s</title></rect>`;
      svg += `<rect x="${left + w1 + w2}" y="${y}" width="${w3}" height="22" fill="#34d399"><title>Dev-clean eval ${num(r.ev)} s</title></rect>`;
      svg += `<text x="${left + w1 + w2 + w3 + 8}" y="${y + 15}" font-family="ui-monospace, monospace" font-size="12" font-weight="700" fill="#e8f0fe">${num(r.total)} s</text>`;
    });
    svg += '</svg>';

    let tbl = '<div class="table-wrap"><table><thead><tr><th>Run (Mac M3 Pro)</th><th class="n">Data wait (s)</th><th class="n">Compute (s)</th><th class="n">Dev-clean eval (s)</th><th class="n">Total (s)</th><th>Provenance</th></tr></thead><tbody>';
    rows.forEach((r) => {
      tbl += `<tr><td>${esc(r.label)}</td><td class="n">${num(r.dw)}</td><td class="n">${num(r.comp)}</td><td class="n">${num(r.ev)}</td><td class="n">${num(r.total)}</td><td>${tagHtml('measured')}</td></tr>`;
    });
    tbl += '</tbody></table></div>';

    let mac = '';
    const hw = (b.hardware_throughput || []).filter((r) => !isNil(r.mac_median_ms));
    if (hw.length) {
      mac = '<h3>Layer micro-benchmarks (Mac M3 Pro)</h3><div class="table-wrap"><table><thead><tr><th>Component</th><th>Precision</th><th class="n">Median (ms)</th><th class="n">Steps / s</th><th class="n">Peak memory (MB)</th><th>Provenance</th></tr></thead><tbody>';
      hw.forEach((r) => {
        mac += `<tr><td>${rich(r.component)}</td><td><code>${esc(r.precision)}</code></td><td class="n">${num(r.mac_median_ms)}</td><td class="n">${num(r.mac_steps_per_sec)}</td><td class="n">${isNil(r.mac_peak_mem_mb) ? '—' : NF.format(r.mac_peak_mem_mb)}</td><td>${tagHtml('measured')}</td></tr>`;
      });
      mac += '</tbody></table></div>';
    }
    openModal('Training time per epoch (Mac M3 Pro)', `<p class="muted small">Measured epoch breakdown and layer micro-benchmarks on Apple M3 Pro (<code>device = mps</code>, fp32) ${tagHtml('measured')}.</p>${svg}${tbl}${mac}`, opener, 'BENCHMARK TELEMETRY · MAC M3 PRO');
  }

  function renderResults() {
    const R = DATA.results, runs = R.runs, s0 = runs.s0, s1 = runs.s1;
    const meta = (s0 && s0.meta) || {};
    const p0 = progress(s0);
    const adStart = adaptiveStartEpoch(s0);

    // Headline telemetry tiles
    const stats = [];
    if (s0 && !isNil(s0.best_dev_eer_pct)) {
      stats.push(`<div class="card corner-card stat"><span class="stat-label">S0 best dev-clean ${term('EER')}</span><span class="stat-value">${num(s0.best_dev_eer_pct)}%</span><span class="stat-sub">at epoch ${s0.best_dev_epoch}${p0 && !p0.done ? ` (epoch ${p0.k}/${p0.total} logged)` : ''} ${tagHtml('measured')}</span></div>`);
    }
    if (!isNil(meta.parameter_count)) {
      stats.push(`<div class="card corner-card stat"><span class="stat-label">Model size</span><span class="stat-value">${fmt(meta.parameter_count)}</span><span class="stat-sub">${term('parameters')} (torchinfo) ${tagHtml('measured')}</span></div>`);
    }
    if (!isNil(meta.dataset_size)) {
      stats.push(`<div class="card corner-card stat"><span class="stat-label">Training data</span><span class="stat-value">${fmt(meta.dataset_size)}</span><span class="stat-sub">utterances · ${fmt(meta.num_speakers)} speakers ${tagHtml('measured')}</span></div>`);
    }
    const adRows = runEpochs(s0).filter((r) => !isNil(r.loss_keep_ratio) && (adStart == null || r.epoch >= adStart));
    if (adRows.length) {
      const a = adRows[0], z = adRows[adRows.length - 1];
      const mn = Math.min.apply(null, adRows.map((r) => r.loss_keep_ratio)), mx = Math.max.apply(null, adRows.map((r) => r.loss_keep_ratio));
      const pts = adRows.map((r, i) => `${(i / Math.max(1, adRows.length - 1) * 200).toFixed(1)},${(20 - ((r.loss_keep_ratio - mn) / (mx - mn || 1)) * 18 - 1).toFixed(1)}`).join(' ');
      stats.push(`<div class="card corner-card stat"><span class="stat-label">${term('AdaptiveDrop')} keep ratio</span><span class="stat-value">${num(a.loss_keep_ratio * 100)}% → ${num(z.loss_keep_ratio * 100)}%</span>` +
        `<span class="stat-sub">epoch ${a.epoch} → epoch ${z.epoch} ${tagHtml('measured')}</span><svg viewBox="0 0 200 22" preserveAspectRatio="none" aria-hidden="true"><polyline fill="none" stroke="${COLORS.s0}" stroke-width="2" points="${pts}"/></svg></div>`);
    }
    $('#stat-row').innerHTML = stats.join('');

    // Charts
    const grid = $('#chart-grid');
    grid.innerHTML = '';
    function addChart(id, titleHtml) {
      const c = document.createElement('article');
      c.className = 'card corner-card chart-card';
      c.innerHTML = `<h3>${titleHtml}</h3><div class="chart" id="${id}"></div>`;
      grid.appendChild(c);
      return $('#' + id, c);
    }
    const compare = s1Comparable(s0, s1);
    const pts = (run, key, scale) => runEpochs(run).filter((r) => !isNil(r[key])).map((r) => ({ x: r.epoch, y: Number(r[key]) * (scale || 1) }));
    const mark = adStart != null ? { x: adStart, label: `AdaptiveDrop starts (epoch ${adStart})` } : null;

    const devSeries = [{ name: 'S0', color: COLORS.s0, points: pts(s0, 'dev_eer_pct') }];
    if (compare) devSeries.push({ name: 'S1', color: COLORS.s1, points: pts(s1, 'dev_eer_pct') });
    lineChart(addChart('chart-dev', `Dev-clean ${term('EER')} per evaluated epoch (%)`), {
      series: devSeries, mark, xMin: 1, yFmt: (v, exact) => (exact ? v.toFixed(2) : v.toFixed(1)) + '%', aria: 'Dev-clean EER per evaluated epoch',
    });

    const lossSeries = [{ name: 'S0', color: COLORS.s0, points: pts(s0, 'train_loss') }];
    if (compare) lossSeries.push({ name: 'S1', color: COLORS.s1, points: pts(s1, 'train_loss') });
    const lossHost = addChart('chart-loss', 'Training loss per epoch' +
      `<button type="button" class="info-btn" aria-label="About training loss comparability" data-tip="From the epoch where AdaptiveDrop starts, S0's training loss is computed on the retained clips only, so it is NOT comparable to S1's. Use the dev-clean EER chart for the S0 vs S1 comparison.">i</button>`);
    lineChart(lossHost, { series: lossSeries, mark, xMin: 1, yFmt: (v) => v.toFixed(2), aria: 'Training loss per epoch' });

    const keep = pts(s0, 'loss_keep_ratio', 100);
    if (keep.length) {
      lineChart(addChart('chart-keep', `${term('AdaptiveDrop')}: share of clips kept (%)`), {
        series: [{ name: 'Keep ratio', color: COLORS.same, points: keep }], mark, xMin: 1, yFmt: (v, exact) => v.toFixed(exact ? 2 : 1) + '%', aria: 'AdaptiveDrop keep ratio per epoch',
      });
      lineChart(addChart('chart-fc', `${term('AdaptiveDrop')}: clips filtered and corrected`), {
        series: [
          { name: 'Filtered', color: COLORS.diff, points: pts(s0, 'loss_num_filtered') },
          { name: 'Corrected', color: COLORS.warn, points: pts(s0, 'loss_num_corrected') },
        ], mark, xMin: 1, yMin: 0, yFmt: (v) => fmt(Math.round(v)), aria: 'Clips filtered and corrected per epoch',
      });
    }

    // Train-vs-dev EER panel: only when a real train-probe EER exists in history.csv.
    const probeRun = [s0, s1].find((r) => runEpochs(r).some((x) => !isNil(x.train_probe_eer_pct)));
    if (probeRun) {
      lineChart(addChart('chart-probe', `Train vs dev ${term('EER')} (%)`), {
        series: [
          { name: 'Train probe EER', color: COLORS.warn, points: pts(probeRun, 'train_probe_eer_pct') },
          { name: 'Dev-clean EER', color: COLORS.s0, points: runEpochs(probeRun).filter((r) => !isNil(r.train_probe_eer_pct) && !isNil(r.dev_eer_pct)).map((r) => ({ x: r.epoch, y: Number(r.dev_eer_pct) })) },
        ], xMin: 1, yFmt: (v, exact) => v.toFixed(exact ? 2 : 1) + '%', aria: 'Train probe EER versus dev-clean EER',
      });
    }
  }

  /* =====================================================================
   * 4. MODELS
   * ===================================================================== */
  const PLAIN_SWITCH = {
    S0: 'Full proposed system: SKA-HNN backbone trained with IAAMSoftmax and AdaptiveDrop.',
    S1: 'Same SKA-HNN backbone trained with standard AAMSoftmax (no AdaptiveDrop).',
    S2: 'Removes sliding-window local-statistics context from the MLASP pooling layer.',
    S3: 'Removes multi-level feature combination (MLFC); pools only the final Fully-TDNN output.',
    S4: 'Removes both MLFC and MLASP local-statistics context (dropped from ablation grid).',
    S5: 'Replaces SKA multi-branch attention in the 2-D CNN with plain 3×3 convolutions.',
    S6: 'Removes the 2-D CNN front end; feeds log-Mel features straight into TDNN-LSTM.',
  };

  function renderModels() {
    const R = DATA.results, a = R.architecture || {};
    const runs = R.runs;
    const switches = a.ablation_switches || [];
    let left = '<div class="card corner-card card-scroll"><span class="micro-label">ABLATION SWITCHES · S0 TO S6</span><h3>SKA-HNN variants (S0–S6)</h3><div class="table-wrap"><table><thead><tr><th>Switch</th><th>What it changes (one-line summary)</th><th class="n">Measured parameters</th><th class="n">Paper parameters</th><th>Status</th></tr></thead><tbody>';
    switches.forEach((s) => {
      const oneLine = PLAIN_SWITCH[s.switch] || clean(s.config_summary);
      const techTip = `${clean(s.config_summary)} (${clean(s.flags)}): ${clean(s.description)}`;
      left += `<tr${s.dropped ? ' class="dim"' : ''}><td><b>${esc(s.switch)}</b>${s.dropped ? ' ' + tagHtml('planned') : ''}</td>` +
        `<td>${explain(esc(oneLine))}<button type="button" class="info-btn" aria-label="Technical details for ${esc(s.switch)}" data-tip="${esc(techTip)}">i</button></td>` +
        `<td class="n">${fmt(s.measured_params)} ${tagHtml(s.measured_label_type)}</td>` +
        `<td class="n">${esc(s.paper_params)} ${tagHtml(s.paper_label_type)}</td>` +
        `<td>${esc(switchStatus(s, runs))}</td></tr>`;
    });
    left += '</tbody></table></div></div>';

    const t0 = (R.tiers || [])[0] || {};
    const refs = (R.benchmarks && R.benchmarks.external_references) || [];
    let right = '<div class="col">';
    (a.baseline_models || []).forEach((m) => {
      right += `<div class="card corner-card model-card"><span class="micro-label">TIER 0 BASELINE ARCHITECTURE</span><h3><span>${rich(m.model_architecture, true)}</span>${tagHtml('measured')}</h3>` +
        `<p class="muted small">A speaker-recognition network used as a baseline (${term('ECAPA-TDNN', 'ECAPA-TDNN')}).</p><dl class="kv">` +
        `<dt>Parameters</dt><dd>${fmt(m.measured_params)}</dd><dt>Channels</dt><dd>${esc(m.channels)}</dd>` +
        `<dt>vs SKA-HNN</dt><dd>${esc(m.delta_vs_skahnn)}</dd>${t0.code_status ? `<dt>Status</dt><dd>${esc(t0.code_status)}</dd>` : ''}</dl></div>`;
    });
    if (refs.length) {
      right += `<div class="card corner-card model-card"><span class="micro-label">PRETRAINED VOXCELEB REFERENCE</span><h3><span>Pretrained SpeechBrain ECAPA-TDNN</span>${tagHtml('external reference')}</h3>` +
        '<p class="muted small">Trained by others on VoxCeleb; shown only as a reference point, not our model.</p>';
      refs.forEach((r) => {
        right += `<dl class="kv"><dt>Evaluated on</dt><dd>${esc(clean(r.dataset).replace(/\s*\(.*$/, ''))}</dd><dt>${term('EER')} [95% CI]</dt><dd>${esc(clean(r.eer))}</dd>` +
          `<dt>${term('minDCF', 'minDCF')} (P=0.05)</dt><dd>${esc(clean(r.mindcf_p05))}</dd></dl>`;
      });
      right += `<p class="muted small">Parameters: ${esc(clean(refs[0].params))} ${tagHtml('external reference')}</p></div>`;
    }
    right += '</div>';
    $('#models-split').innerHTML = left + right;
  }

  /* =====================================================================
   * 5. RIGOR
   * ===================================================================== */
  function renderRigor() {
    const R = DATA.results, meta = (R.runs.s0 && R.runs.s0.meta) || {};
    const refs = (R.benchmarks && R.benchmarks.external_references) || [];
    const hashes = refs.map((r) => {
      const m = /SHA-256:\s*`?([0-9a-f]+)/i.exec(r.dataset || '');
      const name = /(test-clean|test-other)/.exec(r.dataset || '');
      const trials = /\(([\d,]+) trials/.exec(r.dataset || '');
      return m ? `${name ? name[1] : 'trials'} v2${trials ? ' · ' + trials[1] + ' trials' : ''} · SHA-256 ${m[1]}…` : '';
    }).filter(Boolean);
    const ci = refs[0] ? `${clean(refs[0].eer)}` : '';
    const items = [
      ['PROTOCOL 01', 'Full-utterance evaluation', 'Every test clip is scored at its full length, with no cropping.', `Eval mode: full waveform · 16 kHz mono`],
      ['PROTOCOL 02', 'Fixed v2 trial lists', 'Test pairs come from versioned trial lists whose files are fingerprinted with a hash, so everyone scores the same pairs.', hashes.join(' | ')],
      ['PROTOCOL 03', 'dev-clean for model selection only', `The dev-clean set is used only to pick the best checkpoint and the demo's decision ${term('threshold')}.`, `6,200 cross-chapter trials · zero speaker overlap`],
      ['PROTOCOL 04', 'Speaker-level bootstrap intervals', 'Confidence intervals are computed by resampling speakers rather than individual trials.', ci ? `Example (external reference): EER ${ci}` : ''],
      ['PROTOCOL 05', 'Parameters are measured', `Every parameter count on this site comes from torchinfo, never from hand counting.`, !isNil(meta.parameter_count) ? `S0 / S1: ${fmt(meta.parameter_count)} parameters` : ''],
      ['PROTOCOL 06', 'fp32 precision', 'Training and evaluation run in 32-bit floating point.', meta.precision || meta.device ? `Recorded in run_meta.json: precision ${meta.precision || '—'} on ${meta.device || '—'}` : ''],
    ];
    $('#rigor-grid').innerHTML = items.map((it) =>
      `<article class="card corner-card rigor-item"><div class="rigor-head"><span class="micro-label">${esc(it[0])}</span><span class="tick" aria-hidden="true">✓</span></div>` +
      `<h3>${esc(it[1])}</h3><p>${it[2]}</p>` +
      (it[3] ? `<div class="fact">${esc(it[3])}</div>` : '') + '</article>').join('');
  }

  /* =====================================================================
   * Polished "About this project" overlay
   * ===================================================================== */
  const ABOUT_ICONS = {
    scope: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#22d3ee" stroke-width="2" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/></svg>`,
    paper: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#c4b5fd" stroke-width="2" aria-hidden="true"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>`,
    eng: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2" aria-hidden="true"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h.09a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v.09a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>`,
    open: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" stroke-width="2" aria-hidden="true"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`,
  };

  function openAbout(opener) {
    const R = DATA.results;
    const meta = (R && R.runs && R.runs.s0 && R.runs.s0.meta) || {};
    const sw = (R && R.architecture && R.architecture.ablation_switches) || [];
    const s1 = sw.find((s) => s.switch === 'S1'), s5 = sw.find((s) => s.switch === 'S5');
    const parseM = (s) => { const m = /([\d.]+)M/.exec(s || ''); return m ? parseFloat(m[1]) : null; };
    const paperM = (s1 && parseM(s1.paper_params)) || 30.91;
    const oursParams = !isNil(meta.parameter_count) ? Number(meta.parameter_count) : (s1 ? Number(s1.measured_params) : 4167700);
    const scalePct = Math.max(2, Math.min(100, (oursParams / (paperM * 1e6)) * 100));

    let note = '';
    if (s1 && s5) {
      const p1 = parseM(s1.paper_params), p5 = parseM(s5.paper_params);
      const dPaper = p1 != null && p5 != null ? (p1 - p5) : null;
      const dMeas = s1.measured_params - s5.measured_params;
      note = `<div class="open-question-card"><h3>${ABOUT_ICONS.open}Open note: S1 → S5 parameter count</h3><p>The paper's Table II lists S1 at <b>${esc(s1.paper_params)}</b> ${tagHtml('paper')} and S5 (vanilla 2-D CNN) at <b>${esc(s5.paper_params)}</b> ${tagHtml('paper')}` +
        (dPaper != null ? `, a drop of ${dPaper.toFixed(2)}M (${(dPaper / p1 * 100).toFixed(1)}%)` : '') + `. In our torchinfo-audited implementation, replacing the SKA blocks with standard convolutions changes the count from <b>${fmt(s1.measured_params)}</b> (S1 ${tagHtml('measured')}) to <b>${fmt(s5.measured_params)}</b> (S5 ${tagHtml('measured')}), a difference of ${fmt(dMeas)} parameters (${(dMeas / s1.measured_params * 100).toFixed(2)}%) at the scaled width, or about 1.49M at full width. Even removing the whole 2-D CNN front end at full scale removes only about 1.65M in its convolution blocks. So the paper's larger drop between S1 and S5 cannot come from swapping SKA for plain convolutions at fixed channel widths; this is an <b>open question</b> in the published numbers.</p></div>`;
    }

    const html =
      `<div class="about-stats">` +
      `<div class="about-tile"><span class="micro-label">TRAINING SPEAKERS</span><span class="about-tile-val">${fmt(meta.num_speakers)}</span><span class="small muted">LibriSpeech train-clean-100 ${tagHtml('measured')}</span></div>` +
      `<div class="about-tile"><span class="micro-label">TRAINING UTTERANCES</span><span class="about-tile-val">${fmt(meta.dataset_size)}</span><span class="small muted">16 kHz mono FLAC ${tagHtml('measured')}</span></div>` +
      `<div class="about-tile"><span class="micro-label">MODEL PARAMETERS</span><span class="about-tile-val">${fmt(oursParams)}</span><span class="small muted">torchinfo audited ${tagHtml('measured')}</span></div>` +
      `</div>` +
      `<div class="about-scale-card">` +
      `<h3>${ABOUT_ICONS.scope}Dataset scope &amp; model scale comparison</h3>` +
      `<p>Training uses LibriSpeech <code>train-clean-100</code>${!isNil(meta.num_speakers) ? `: ${fmt(meta.num_speakers)} speakers and ${fmt(meta.dataset_size)} utterances ${tagHtml('measured')}` : ''}. ` +
      `The network widths are scaled down (C=32, C1=256) from the paper's VoxCeleb2 setup (5,994 speakers, C=128, C1=512, 30.91M parameters ${tagHtml('paper')})` +
      `${!isNil(meta.parameter_count) ? `; this model has ${fmt(meta.parameter_count)} parameters ${tagHtml('measured')}` : ''}, to suit a small speaker pool and a laptop. ` +
      `The test sets each contain 29 qualifying multi-chapter speakers, so confidence intervals are roughly ±1.0% wide.</p>` +
      `<div class="scale-bars">` +
      `<div class="scale-row"><span>Paper full-scale (C=128)</span><div class="scale-track"><div class="scale-fill-paper" style="width:100%"></div></div><span class="n">30.91M ${tagHtml('paper')}</span></div>` +
      `<div class="scale-row"><span>This project (C=32)</span><div class="scale-track"><div class="scale-fill-ours" style="width:${scalePct.toFixed(2)}%"></div></div><span class="n">${fmt(oursParams)} ${tagHtml('measured')}</span></div>` +
      `</div></div>` +
      `<div class="two-col">` +
      `<div class="about-card"><h3>${ABOUT_ICONS.paper}From the paper</h3><ul>` +
      `<li>${tagHtml('paper')} 80-dimensional log-Mel input (25 ms window, 10 ms hop), 180-frame training segments, batches of 64 distinct speakers.</li>` +
      `<li>${tagHtml('paper')} 2-D CNN → TDNN-LSTM → Fully-TDNN → MLFC → MLASP → two fully connected layers (192-dimensional embedding).</li>` +
      `<li>${tagHtml('paper')} Loss settings s=30, m=0.2, AdaptiveDrop threshold 0.423, filtering from epoch 5.</li></ul></div>` +
      `<div class="about-card"><h3>${ABOUT_ICONS.eng}Engineering choices of this project</h3><ul>` +
      `<li>Scaled widths (C=32, C1=256, attention width 128).</li>` +
      `<li>Label-correction start at epoch 5, and a safeguard that keeps at least half of each batch.</li>` +
      `<li>Local statistics in fp32 with a small variance floor; replicate padding.</li>` +
      `<li>On-the-fly babble and colored-noise augmentation in place of MUSAN; reverberation switched off on the Mac.</li>` +
      `<li>Taken from outside the paper and not verified: the label-correction threshold 0.5, the SKA branch dilations (1 and 2) and reduction ratio (8), and the local-statistics window of 21 frames.</li></ul></div>` +
      `</div>` +
      (note || '');
    openModal('About this project', html, opener, 'PROJECT SPECIFICATION & PROVENANCE');
  }

  /* =====================================================================
   * Boot
   * ===================================================================== */
  $('#about-btn').addEventListener('click', (e) => {
    if (DATA.results) openAbout(e.currentTarget);
    else openModal('About this project', '<p class="muted">Project data is still loading.</p>', e.currentTarget, 'PROJECT SPECIFICATION & PROVENANCE');
  });
  $('#btn-view-data').addEventListener('click', (e) => { if (DATA.results) openViewData(e.currentTarget); });
  $('#btn-view-timing').addEventListener('click', (e) => { if (DATA.results) openTiming(e.currentTarget); });

  function safe(fn, hostSel) {
    try { fn(); } catch (err) {
      console.error(err);
      if (hostSel) $(hostSel).innerHTML = '<p class="muted">This section could not be drawn from the data currently available.</p>';
    }
  }

  async function boot() {
    const [info, results] = await Promise.allSettled([
      fetch('/api/info').then((r) => { if (!r.ok) throw new Error('info'); return r.json(); }),
      fetch('/api/results').then((r) => { if (!r.ok) throw new Error('results'); return r.json(); }),
    ]);
    DATA.info = info.status === 'fulfilled' ? info.value : null;
    DATA.results = results.status === 'fulfilled' ? results.value : null;
    safe(renderDemoFacts);
    if (!DATA.results) {
      ['#pipeline', '#chart-grid', '#models-split', '#rigor-grid'].forEach((s) => {
        $(s).innerHTML = '<p class="muted">Project data could not be loaded. Please refresh the page.</p>';
      });
      return;
    }
    safe(renderPipeline, '#pipeline');
    safe(renderResults, '#chart-grid');
    safe(renderModels, '#models-split');
    safe(renderRigor, '#rigor-grid');
    const qOverlay = new URLSearchParams(location.search).get('overlay');
    if (qOverlay === 'about') openAbout($('#about-btn'));
    else if (qOverlay === 'timing') openTiming($('#btn-view-timing'));
    else if (qOverlay === 'data') openViewData($('#btn-view-data'));
    else if (qOverlay === 'stage') {
      const st = $$('.stage-card')[2] || $$('.stage-card')[0];
      if (st) st.click();
    }
  }

  activateTab(location.hash.replace('#', '') || 'demo', { updateHash: false });
  boot();
})();
