'use strict';
/* ── scenario.js — Scenario detail page: advantage heatmap, metric charts,
   seed selector, per-seed table. ─────────────────────────────────────── */

// ── Constants ──────────────────────────────────────────────────────────────
const FT_COLOR      = '#f97316';
const RL_COLOR      = '#10b981';
const GRID_COLOR    = 'rgba(255,255,255,.05)';
const TICK_COLOR    = 'rgba(255,255,255,.3)';
const TXT_COLOR     = '#94a3b8';

const METRICS = ['waiting_time', 'queue_length', 'stopped_ratio', 'throughput'];
const METRIC_INFO = {
  waiting_time:  { label: 'Wait Time',    unit: 's',       maximize: false, precision: 2 },
  queue_length:  { label: 'Queue',        unit: 'veh',     maximize: false, precision: 2 },
  stopped_ratio: { label: 'Stop Ratio',   unit: '',        maximize: false, precision: 3 },
  throughput:    { label: 'Throughput',   unit: 'veh/st',  maximize: true,  precision: 1 },
};

const SCENARIO_ICONS = {
  bottleneck: '🔀', main: '🛣️', pedestrian: '🚶', hexagon: '⬡',
};

// ── State ──────────────────────────────────────────────────────────────────
let manifest     = null;
let currentData  = null;
let allSeedData  = {};    // seed → data (cached)
let scenario     = null;
let currentSeed  = null;
const chartInstances = {};

// ── Bootstrap ──────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  // Parse ?s=<scenario> from URL
  const params = new URLSearchParams(window.location.search);
  scenario     = params.get('s') || 'main';

  try {
    manifest = await fetch('data/manifest.json').then(r => {
      if (!r.ok) throw new Error('manifest missing');
      return r.json();
    });
    const scData = manifest.scenarios[scenario];
    if (!scData) throw new Error(`Unknown scenario: ${scenario}`);

    document.title = `${scData.title} — MARL-ATSC Baku`;
    renderHeader(scData);
    renderSeedTabs(scData.seeds);

    // Load first seed and pre-cache remaining in background
    currentSeed = scData.seeds[0];
    await loadAndRender(currentSeed);
    prefetchRemaining(scData.seeds);
    renderSeedsTable(scData);
  } catch(e) {
    console.error(e);
    showError(e.message);
  }
});

// ── Header ─────────────────────────────────────────────────────────────────
function renderHeader(sc) {
  document.getElementById('sc-icon').textContent = SCENARIO_ICONS[scenario] || '🚦';
  document.getElementById('sc-title').textContent = sc.title;
  document.getElementById('sc-subtitle').textContent = sc.subtitle;
  document.getElementById('sc-desc').textContent = sc.description;

  const pills = document.getElementById('sc-pills');
  pills.innerHTML = [
    `${sc.n_tls} Traffic Light${sc.n_tls > 1 ? 's' : ''}`,
    `${sc.demand_veh_h?.toLocaleString()} veh/h`,
    sc.tl_ids?.join(', '),
    `${sc.seeds?.length} eval seeds`,
  ].map(t => `<span class="sc-pill">${t}</span>`).join('');
}

// ── Seed tabs ──────────────────────────────────────────────────────────────
function renderSeedTabs(seeds) {
  const bar = document.getElementById('seed-bar');
  seeds.forEach(seed => {
    const btn = document.createElement('button');
    btn.className = 'seed-tab' + (seed === currentSeed ? ' active' : '');
    btn.textContent = `Seed ${seed}`;
    btn.onclick = () => switchSeed(seed);
    bar.appendChild(btn);
  });
}

function setActiveTab(seed) {
  document.querySelectorAll('.seed-tab').forEach(btn => {
    btn.classList.toggle('active', btn.textContent === `Seed ${seed}`);
  });
}

async function switchSeed(seed) {
  if (seed === currentSeed) return;
  currentSeed = seed;
  setActiveTab(seed);
  await loadAndRender(seed);
}

// ── Data loading ───────────────────────────────────────────────────────────
async function loadAndRender(seed) {
  if (!allSeedData[seed]) {
    try {
      allSeedData[seed] = await fetch(`data/${scenario}_s${seed}.json`).then(r => {
        if (!r.ok) throw new Error(`data file for seed ${seed} not found`);
        return r.json();
      });
    } catch(e) {
      console.error(e);
      showError(location.protocol === 'file:'
        ? 'Opened as a file:// URL — the browser blocks loading the data. Serve over HTTP: "cd website && python -m http.server 8000", then open http://localhost:8000.'
        : `Data for seed ${seed} not found. From the repo root run python generate_site_data.py, then refresh.`);
      return;
    }
  }
  currentData = allSeedData[seed];
  renderAll(currentData);
}

async function prefetchRemaining(seeds) {
  const remaining = seeds.filter(s => s !== currentSeed);
  for (const seed of remaining) {
    if (!allSeedData[seed]) {
      try {
        allSeedData[seed] = await fetch(`data/${scenario}_s${seed}.json`).then(r => r.json());
      } catch(_) { /* silently skip unavailable seeds */ }
    }
  }
}

// ── Render all visualizations ──────────────────────────────────────────────
function renderAll(data) {
  renderMetricCards(data);
  renderHeatmap(data);
  renderCharts(data);
}

// ── Metric summary cards ───────────────────────────────────────────────────
function renderMetricCards(data) {
  const container = document.getElementById('metric-cards');
  container.innerHTML = '';
  const sum = data.summary;

  METRICS.forEach(m => {
    const info = METRIC_INFO[m];
    const imp  = sum.improvement[m];
    const pos  = imp >= 0;
    const ft   = sum.fixed_time[m];
    const rl   = sum.ppo_best[m];
    const u    = info.unit ? ` ${info.unit}` : '';

    const mc = document.createElement('div');
    mc.className = `mc ${pos ? 'pos-win' : 'neg-win'}`;
    mc.innerHTML = `
      <div class="mc-label">${info.label}</div>
      <div class="mc-imp ${pos ? 'pos' : 'neg'}">${pos ? '+' : ''}${imp.toFixed(1)}%</div>
      <div class="mc-row">
        <span>Fixed-Time</span>
        <span class="mc-ft">${ft.toFixed(info.precision)}${u}</span>
      </div>
      <div class="mc-row">
        <span>PPO Best</span>
        <span class="mc-rl">${rl.toFixed(info.precision)}${u}</span>
      </div>
    `;
    container.appendChild(mc);
  });
}

// ── Advantage Heatmap ──────────────────────────────────────────────────────
let heatmapData = null;  // saved for tooltip

function renderHeatmap(data) {
  heatmapData = data;
  const canvas = document.getElementById('heatmap');
  if (!canvas) return;

  const dpr    = window.devicePixelRatio || 1;
  const cw     = canvas.offsetWidth  || canvas.parentElement.offsetWidth;
  const ch     = 260;
  canvas.width  = cw * dpr;
  canvas.height = ch * dpr;
  canvas.style.width  = cw  + 'px';
  canvas.style.height = ch  + 'px';

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  drawHeatmap(ctx, data, cw, ch);

  // Tooltip on hover
  canvas.onmousemove = e => heatmapHover(e, canvas, data, cw, ch);
  canvas.onmouseleave = () => {
    document.getElementById('heatmap-tooltip').classList.remove('visible');
  };
}

function drawHeatmap(ctx, data, W, H) {
  const metrics   = METRICS;
  const nMetrics  = metrics.length;
  const warmup    = data.warmup || 30;
  const labelW    = 110;
  const timeAxisH = 30;
  const rowH      = (H - timeAxisH) / nMetrics;
  const n         = Math.min(data.fixed_time.waiting_time.length,
                             data.ppo_best.waiting_time.length);
  const stepW     = (W - labelW) / n;

  // Background
  ctx.fillStyle = 'rgba(14,21,38,0)';
  ctx.fillRect(0, 0, W, H);

  metrics.forEach((metric, mi) => {
    const info   = METRIC_INFO[metric];
    const ft     = data.fixed_time[metric];
    const rl     = data.ppo_best[metric];
    const range  = Math.max(1e-9, ...ft.slice(0, n), ...rl.slice(0, n));

    const y0 = mi * rowH;

    // Row background separator
    if (mi % 2 === 0) {
      ctx.fillStyle = 'rgba(255,255,255,.015)';
      ctx.fillRect(0, y0, W, rowH);
    }

    // Draw per-step cells
    for (let t = 0; t < n; t++) {
      const ftV = ft[t] ?? 0;
      const rlV = rl[t] ?? 0;
      // improvement: positive = RL better
      const raw = info.maximize
        ? (rlV - ftV) / range
        : (ftV - rlV) / range;
      const imp  = raw;
      const warm = t < warmup;
      const base = warm ? 0.18 : 0.85;
      const intensity = Math.min(1, Math.abs(imp) * 4);

      if (Math.abs(imp) < 0.01) {
        ctx.fillStyle = `rgba(148,163,184,${base * 0.12})`;
      } else if (imp > 0) {
        ctx.fillStyle = `rgba(16,185,129,${base * intensity})`;
      } else {
        ctx.fillStyle = `rgba(239,68,68,${base * intensity})`;
      }

      const x = labelW + t * stepW;
      ctx.fillRect(x, y0 + 1, Math.max(1, stepW - 0.3), rowH - 2);
    }

    // Label (left panel)
    ctx.fillStyle = 'rgba(14,21,38,0.8)';
    ctx.fillRect(0, y0, labelW - 4, rowH);
    ctx.fillStyle = TXT_COLOR;
    ctx.font = '11.5px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(info.label, 8, y0 + rowH / 2);

    // Row border bottom
    ctx.strokeStyle = 'rgba(255,255,255,.06)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, y0 + rowH);
    ctx.lineTo(W, y0 + rowH);
    ctx.stroke();
  });

  // Warmup divider line
  const warmupX = labelW + (data.warmup || 30) * stepW;
  ctx.strokeStyle = 'rgba(255,255,255,.65)';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(warmupX, 0);
  ctx.lineTo(warmupX, nMetrics * rowH);
  ctx.stroke();
  ctx.setLineDash([]);

  // Time axis labels
  const axisY = nMetrics * rowH;
  ctx.fillStyle = TXT_COLOR;
  ctx.font = '10.5px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';

  ctx.fillStyle = 'rgba(255,255,255,.35)';
  ctx.font = '10px Inter, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('warm-up', labelW + 4, axisY + 6);
  ctx.textAlign = 'center';
  ctx.fillStyle = TXT_COLOR;

  const ticks = [0, Math.floor(n * 0.25), Math.floor(n * 0.5), Math.floor(n * 0.75), n - 1];
  ticks.forEach(t => {
    const x = labelW + (t + 0.5) * stepW;
    ctx.fillStyle = 'rgba(255,255,255,.25)';
    ctx.fillRect(x, axisY, 1, 5);
    ctx.fillStyle = TXT_COLOR;
    ctx.fillText(`t=${t}`, x, axisY + 8);
  });
}

// ── Heatmap hover tooltip ──────────────────────────────────────────────────
function heatmapHover(e, canvas, data, W, H) {
  const rect    = canvas.getBoundingClientRect();
  const mx      = e.clientX - rect.left;
  const my      = e.clientY - rect.top;
  const labelW  = 110;
  const nM      = METRICS.length;
  const timeAxisH = 30;
  const rowH    = (H - timeAxisH) / nM;
  const n       = data.fixed_time.waiting_time.length;
  const stepW   = (W - labelW) / n;
  const tt      = document.getElementById('heatmap-tooltip');

  if (mx < labelW || my > nM * rowH) {
    tt.classList.remove('visible');
    return;
  }

  const t  = Math.floor((mx - labelW) / stepW);
  const mi = Math.floor(my / rowH);
  if (t < 0 || t >= n || mi < 0 || mi >= nM) {
    tt.classList.remove('visible');
    return;
  }

  const metric = METRICS[mi];
  const info   = METRIC_INFO[metric];
  const ft     = data.fixed_time[metric][t] ?? 0;
  const rl     = data.ppo_best[metric][t]   ?? 0;
  const imp    = info.maximize
    ? (ft === 0 ? 0 : (rl - ft) / ft * 100)
    : (ft === 0 ? 0 : (ft - rl) / ft * 100);
  const pos    = imp >= 0;
  const u      = info.unit ? ` ${info.unit}` : '';
  const warm   = t < (data.warmup || 30);

  tt.innerHTML = `
    <div class="tt-step">Step ${t}${warm ? ' <em>(warm-up)</em>' : ''}</div>
    <div class="tt-row"><span class="tt-label">${info.label} FT</span><span class="tt-ft">${ft.toFixed(info.precision)}${u}</span></div>
    <div class="tt-row"><span class="tt-label">${info.label} RL</span><span class="tt-rl">${rl.toFixed(info.precision)}${u}</span></div>
    <div class="tt-row"><span class="tt-label">Improvement</span><span class="tt-imp ${pos ? 'pos':'neg'}">${pos?'+':''}${imp.toFixed(1)}%</span></div>
  `;

  // Position tooltip so it stays inside container
  const cw = canvas.offsetWidth;
  const xPos = mx + 16 + 180 > cw ? mx - 196 : mx + 16;
  const yPos = Math.max(0, my - 10);
  tt.style.left = xPos + 'px';
  tt.style.top  = yPos + 'px';
  tt.classList.add('visible');
}

// ── Metric time-series charts ──────────────────────────────────────────────
function renderCharts(data) {
  METRICS.forEach(m => buildMetricChart(m, data));
}

function buildMetricChart(metric, data) {
  const info   = METRIC_INFO[metric];
  const ft     = data.fixed_time[metric];
  const rl     = data.ppo_best[metric];
  const warmup = data.warmup || 30;
  const n      = Math.min(ft.length, rl.length);
  const labels = Array.from({length: n}, (_, i) => i);

  // Update improvement badge
  const imp    = data.summary.improvement[metric];
  const impEl  = document.getElementById(`imp-${metric}`);
  if (impEl) {
    const pos = imp >= 0;
    impEl.className = `chart-imp ${pos ? 'pos' : 'neg'}`;
    impEl.textContent = `${pos ? '+' : ''}${imp.toFixed(1)}%`;
  }

  const canvas = document.getElementById(`chart-${metric}`);
  if (!canvas) return;
  if (chartInstances[metric]) { chartInstances[metric].destroy(); }

  // Determine fill colors based on whether metric is maximize or minimize
  // above = RL line is ABOVE FT line; below = RL is BELOW FT
  // For minimize: below FT = RL better = green; above FT = RL worse = red
  // For maximize: above FT = RL better = green; below FT = RL worse = red
  const aboveColor = info.maximize ? 'rgba(16,185,129,.18)' : 'rgba(239,68,68,.18)';
  const belowColor = info.maximize ? 'rgba(239,68,68,.18)' : 'rgba(16,185,129,.18)';

  // Warmup background plugin
  const warmupPlugin = {
    id: `warmup-${metric}`,
    beforeDraw(chart) {
      const { ctx, chartArea: a, scales } = chart;
      if (!a) return;
      const xWarmup = scales.x.getPixelForValue(warmup);
      ctx.save();
      ctx.fillStyle = 'rgba(255,255,255,.03)';
      ctx.fillRect(a.left, a.top, xWarmup - a.left, a.height);
      // warmup divider
      ctx.strokeStyle = 'rgba(255,255,255,.25)';
      ctx.lineWidth   = 1;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(xWarmup, a.top);
      ctx.lineTo(xWarmup, a.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    },
  };

  chartInstances[metric] = new Chart(canvas, {
    type: 'line',
    plugins: [warmupPlugin],
    data: {
      labels,
      datasets: [
        {
          label: 'Fixed-Time',
          data: ft.slice(0, n),
          borderColor: FT_COLOR,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.25,
          fill: false,
          order: 2,
        },
        {
          label: 'PPO (Best)',
          data: rl.slice(0, n),
          borderColor: RL_COLOR,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.25,
          fill: { target: '-1', above: aboveColor, below: belowColor },
          order: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 600 },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(14,21,38,.95)',
          borderColor: 'rgba(255,255,255,.12)',
          borderWidth: 1,
          titleColor: TXT_COLOR,
          bodyColor: '#f0f4ff',
          padding: 10,
          callbacks: {
            title: ctx => `Step ${ctx[0].label}${ctx[0].label < warmup ? ' (warm-up)' : ''}`,
            label: ctx => {
              const u = info.unit ? ` ${info.unit}` : '';
              const color = ctx.datasetIndex === 0 ? '🟠' : '🟢';
              return ` ${color} ${ctx.dataset.label}: ${ctx.raw?.toFixed(info.precision)}${u}`;
            },
          },
        },
      },
      scales: {
        x: {
          grid:  { color: GRID_COLOR },
          ticks: { color: TXT_COLOR, font: { size: 10 }, maxTicksLimit: 8,
                    callback: v => `t${v}` },
        },
        y: {
          grid:  { color: GRID_COLOR },
          ticks: { color: TXT_COLOR, font: { size: 10 },
                    callback: v => info.precision === 3 ? v.toFixed(3) : v.toFixed(1) },
          beginAtZero: true,
        },
      },
    },
  });
}

// ── Per-seed breakdown table ───────────────────────────────────────────────
function renderSeedsTable(scData) {
  const wrap = document.getElementById('seeds-table-wrap');
  if (!wrap || !scData.seed_summaries) return;

  const summs = scData.seed_summaries;

  let headerCols = METRICS.map(m =>
    `<th>FT ${METRIC_INFO[m].label}</th><th>RL ${METRIC_INFO[m].label}</th><th>Δ</th>`
  ).join('');

  let rows = summs.map(s => {
    const cols = METRICS.map(m => {
      const ft  = s.fixed_time[m];
      const rl  = s.ppo_best[m];
      const imp = s.improvement[m];
      const pos = imp >= 0;
      const u   = METRIC_INFO[m].unit ? ` ${METRIC_INFO[m].unit}` : '';
      const p   = METRIC_INFO[m].precision;
      return `
        <td class="ft-v">${ft.toFixed(p)}${u}</td>
        <td class="rl-v">${rl.toFixed(p)}${u}</td>
        <td class="imp-v ${pos ? 'pos' : 'neg'}">${pos?'+':''}${imp.toFixed(1)}%</td>
      `;
    }).join('');
    return `<tr><td class="seed-num">${s.seed}</td>${cols}</tr>`;
  }).join('');

  // Average row
  const avgCols = METRICS.map(m => {
    const imp = scData.mean_improvement[m];
    const pos = imp >= 0;
    return `<td class="ft-v">—</td><td class="rl-v">—</td>
            <td class="imp-v ${pos?'pos':'neg'}">${pos?'+':''}${imp.toFixed(1)}%</td>`;
  }).join('');
  rows += `<tr class="avg-row"><td class="seed-num">mean</td>${avgCols}</tr>`;

  wrap.innerHTML = `
    <div style="overflow-x:auto">
      <table class="seeds-table">
        <thead>
          <tr>
            <th>Seed</th>
            ${headerCols}
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <p style="font-size:11px;color:var(--txt-3);margin-top:10px">
      FT = fixed-time baseline · RL = PPO best checkpoint · Δ = % improvement · Positive Δ = RL better
    </p>
  `;
}

// ── Error display ──────────────────────────────────────────────────────────
function showError(msg) {
  ['metric-cards', 'seeds-table-wrap'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = `
      <div class="loading-state" style="height:80px;flex-direction:column;gap:6px">
        <div style="font-size:14px;font-weight:600;color:var(--red)">Error loading data</div>
        <div style="font-size:12px;color:var(--txt-3)">${msg}</div>
      </div>`;
  });
}
