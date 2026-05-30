'use strict';
/* ── landing.js — Landing page: loads manifest, renders scenario cards,
   mini sparkline charts, hero stats, and results table. ───────────────── */

const SCENARIO_ICONS = {
  bottleneck: '🔀',
  main:       '🛣️',
  pedestrian: '🚶',
  hexagon:    '⬡',
};

const METRIC_LABELS = {
  waiting_time:  'Avg. Wait Time',
  queue_length:  'Queue Length',
  stopped_ratio: 'Stopped Ratio',
  throughput:    'Throughput',
};

const FT_COLOR  = '#f97316';
const RL_COLOR  = '#10b981';

let manifest = null;
// mini chart instances keyed by scenario (for cleanup)
const miniCharts = {};

/* ── Bootstrap ─────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', async () => {
  try {
    manifest = await fetch('data/manifest.json').then(r => {
      if (!r.ok) throw new Error('manifest not found');
      return r.json();
    });
    renderHeroStats();
    renderScenarioCards();
    renderResultsTable();
    initScrollFadeIn();
  } catch (e) {
    console.error('Failed to load manifest:', e);
    showDataError();
  }
});

/* ── Hero stats ─────────────────────────────────────────────────────────── */
function renderHeroStats() {
  document.querySelectorAll('.hero-stat .num').forEach(el => {
    const sc     = el.dataset.target;
    const metric = el.dataset.metric;
    const sc_data = manifest.scenarios[sc];
    if (!sc_data) return;
    const val = sc_data.mean_improvement[metric];
    if (val === undefined) return;
    animateCounter(el, val, '%', val >= 0 ? '+' : '');
  });
}

function animateCounter(el, target, suffix = '', prefix = '') {
  const duration = 1400;
  const start = performance.now();
  const ease   = t => 1 - Math.pow(1 - t, 3);

  function frame(now) {
    const t   = Math.min((now - start) / duration, 1);
    const val = target * ease(t);
    el.textContent = prefix + val.toFixed(1) + suffix;
    if (t < 1) requestAnimationFrame(frame);
    else       el.textContent = prefix + target.toFixed(1) + suffix;
  }
  requestAnimationFrame(frame);
}

/* ── Scenario cards ─────────────────────────────────────────────────────── */
function renderScenarioCards() {
  const grid = document.getElementById('scenario-cards');
  grid.innerHTML = '';

  const scenarios = ['bottleneck', 'main', 'pedestrian', 'hexagon'];
  scenarios.forEach(sc => {
    const data = manifest.scenarios[sc];
    if (!data) return;
    const imp  = data.mean_improvement;
    const card = document.createElement('div');
    card.className = 'scenario-card fade-in';
    card.innerHTML = `
      <div class="card-header">
        <div>
          <div class="card-title">${data.title}</div>
          <div class="card-subtitle">${data.subtitle}</div>
        </div>
        <div class="card-badge">${SCENARIO_ICONS[sc]} ${sc}</div>
      </div>

      <div class="card-meta">
        <span><strong>${data.n_tls}</strong> Traffic Light${data.n_tls > 1 ? 's' : ''}</span>
        <span><strong>${data.demand_veh_h?.toLocaleString()}</strong> veh/h</span>
        <span><strong>${data.seeds?.length}</strong> eval seeds</span>
      </div>

      <div class="card-chart-wrap">
        <canvas id="mini-${sc}" style="width:100%;height:100%"></canvas>
      </div>

      <div class="card-metrics">
        <div class="card-metric">
          <div class="m-label">Wait Time ↓</div>
          <div class="m-val">${fmtImp(imp.waiting_time)}</div>
          <div class="m-sub">vs fixed-time</div>
        </div>
        <div class="card-metric">
          <div class="m-label">Queue ↓</div>
          <div class="m-val">${fmtImp(imp.queue_length)}</div>
          <div class="m-sub">vs fixed-time</div>
        </div>
        <div class="card-metric">
          <div class="m-label">Stop Ratio ↓</div>
          <div class="m-val">${fmtImp(imp.stopped_ratio)}</div>
          <div class="m-sub">vs fixed-time</div>
        </div>
        <div class="card-metric">
          <div class="m-label">Throughput ↑</div>
          <div class="m-val">${fmtImp(imp.throughput)}</div>
          <div class="m-sub">vs fixed-time</div>
        </div>
      </div>

      <a class="card-link" href="scenario.html?s=${sc}">
        View step-by-step evaluation <span style="font-size:16px">→</span>
      </a>
    `;
    grid.appendChild(card);
  });

  initScrollFadeIn();

  // After cards are in DOM, load seed-42 data for mini sparklines
  scenarios.forEach(sc => loadMiniChart(sc));
}

async function loadMiniChart(sc) {
  const seeds = manifest.scenarios[sc]?.seeds || [];
  const seed  = seeds.includes(42) ? 42 : seeds[0];
  try {
    const data = await fetch(`data/${sc}_s${seed}.json`).then(r => r.json());
    renderMiniChart(sc, data);
  } catch (e) {
    console.warn(`Mini chart for ${sc} not ready yet (data still generating)`);
    // Render placeholder text
    const canvas = document.getElementById(`mini-${sc}`);
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = 'rgba(148,163,184,0.3)';
      ctx.font = '12px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Data generating…', canvas.width / 2, canvas.height / 2);
    }
  }
}

function renderMiniChart(sc, data) {
  const canvas = document.getElementById(`mini-${sc}`);
  if (!canvas) return;
  if (miniCharts[sc]) { miniCharts[sc].destroy(); }

  const warmup = data.warmup || 30;
  const ft = data.fixed_time.waiting_time;
  const rl = data.ppo_best.waiting_time;
  const n  = Math.min(ft.length, rl.length);
  const labels = Array.from({length: n}, (_, i) => i);

  miniCharts[sc] = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          data: ft,
          borderColor: FT_COLOR,
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.3,
          fill: false,
        },
        {
          data: rl,
          borderColor: RL_COLOR,
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.3,
          fill: { target: '-1', above: 'rgba(239,68,68,.15)', below: 'rgba(16,185,129,.15)' },
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 800 },
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: {
        x: { display: false },
        y: { display: false, beginAtZero: true },
      },
    },
  });
}

/* ── Results table ──────────────────────────────────────────────────────── */
function renderResultsTable() {
  const wrap = document.getElementById('results-table-wrap');
  const scenarios = ['bottleneck', 'main', 'pedestrian', 'hexagon'];

  let rows = '';
  scenarios.forEach(sc => {
    const d   = manifest.scenarios[sc];
    const imp = d?.mean_improvement || {};
    rows += `
      <tr>
        <td>
          <div class="scenario-name">${SCENARIO_ICONS[sc]} ${d?.title || sc}</div>
          <div class="tl-count">${d?.n_tls} TL${d?.n_tls > 1 ? 's' : ''} · ${d?.demand_veh_h?.toLocaleString()} veh/h · ${d?.seeds?.length} seeds</div>
        </td>
        <td><span class="imp ${imp.waiting_time >= 0 ? 'pos' : 'neg'}">${fmtImp(imp.waiting_time)}</span></td>
        <td><span class="imp ${imp.queue_length >= 0 ? 'pos' : 'neg'}">${fmtImp(imp.queue_length)}</span></td>
        <td><span class="imp ${imp.stopped_ratio >= 0 ? 'pos' : 'neg'}">${fmtImp(imp.stopped_ratio)}</span></td>
        <td><span class="imp ${imp.throughput >= 0 ? 'pos' : 'neg'}">${fmtImp(imp.throughput)}</span></td>
        <td><a href="scenario.html?s=${sc}" class="btn btn-outline" style="padding:6px 14px;font-size:12px">Details →</a></td>
      </tr>
    `;
  });

  // Average row
  const metrics = ['waiting_time', 'queue_length', 'stopped_ratio', 'throughput'];
  const avgs    = {};
  metrics.forEach(m => {
    const vals = scenarios.map(sc => manifest.scenarios[sc]?.mean_improvement[m] || 0);
    avgs[m] = vals.reduce((a, b) => a + b, 0) / vals.length;
  });

  rows += `
    <tr style="font-weight:700; background:var(--bg-elevated)">
      <td>Cross-scenario average</td>
      ${metrics.map(m => `<td><span class="imp ${avgs[m] >= 0 ? 'pos' : 'neg'}">${fmtImp(avgs[m])}</span></td>`).join('')}
      <td></td>
    </tr>
  `;

  wrap.innerHTML = `
    <table class="results-table">
      <thead>
        <tr>
          <th>Scenario</th>
          <th>Wait Time ↓</th>
          <th>Queue Length ↓</th>
          <th>Stop Ratio ↓</th>
          <th>Throughput ↑</th>
          <th></th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
    <p style="font-size:12px;color:var(--txt-3);margin-top:12px;text-align:right">
      Positive % = RL better. Evaluation window: ${manifest.steps} steps, ${manifest.warmup} warm-up discarded.
      Using <strong>ppo_best.zip</strong> checkpoint (best waiting-time improvement during training).
    </p>
  `;
}

/* ── Helpers ─────────────────────────────────────────────────────────────── */
function fmtImp(v) {
  if (v === undefined || v === null) return '—';
  const s = v >= 0 ? '+' : '';
  return `${s}${v.toFixed(1)}%`;
}

function showDataError() {
  const grid = document.getElementById('scenario-cards');
  if (grid) {
    const viaFile = location.protocol === 'file:';
    grid.innerHTML = `
      <div class="loading-state" style="grid-column:1/-1;flex-direction:column;gap:8px">
        <div style="font-size:15px;font-weight:600">Could not load data/manifest.json</div>
        <div style="font-size:13px;color:var(--txt-3);max-width:520px;text-align:center;line-height:1.6">
          ${viaFile
            ? `This page was opened as a <code style="background:var(--bg-elevated);padding:2px 6px;border-radius:4px">file://</code> URL, so the browser blocks loading the data. Serve it over HTTP instead: <code style="background:var(--bg-elevated);padding:2px 6px;border-radius:4px">cd website &amp;&amp; python -m http.server 8000</code>, then open <code style="background:var(--bg-elevated);padding:2px 6px;border-radius:4px">http://localhost:8000</code>.`
            : `The data files were not found. From the repo root run <code style="background:var(--bg-elevated);padding:2px 6px;border-radius:4px">python generate_site_data.py</code>, then refresh.`}
        </div>
      </div>
    `;
  }
  const rt = document.getElementById('results-table-wrap');
  if (rt) rt.innerHTML = '';
  document.querySelectorAll('.hero-stat .num').forEach(el => el.textContent = '—');
}

/* ── Scroll fade-in ─────────────────────────────────────────────────────── */
function initScrollFadeIn() {
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) { e.target.classList.add('visible'); obs.unobserve(e.target); }
    });
  }, { threshold: 0.1 });
  document.querySelectorAll('.fade-in').forEach(el => obs.observe(el));
}
