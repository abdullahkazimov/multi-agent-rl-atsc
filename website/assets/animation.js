'use strict';
/* ── animation.js ─────────────────────────────────────────────────────────
   Traffic animation: real SUMO vehicle positions + per-TL congestion data.

   Vehicle positions are captured from TraCI at every decision step and
   stored in anim_{scenario}_s{seed}.json as flat arrays:
     [id, x_sumo, y_sumo, spd_kmh,  id, x_sumo, …]
   id = adler32 hash of SUMO vehicle ID (16-bit) — used for cross-step
   interpolation so vehicles glide smoothly between snapshots.

   When PAUSED: step does not advance → vehicles freeze at exact captured
   positions. No synthetic particle movement ever runs independently.
   ─────────────────────────────────────────────────────────────────────── */

// ── Edge connectivity (topology only — positions come from data.network) ──
const EDGES = {
  bottleneck: [
    { from:'TL_Hasan_Aliyev', to:'TL_Salamzade' },
    { from:'TL_Salamzade',    to:'TL_Hasan_Aliyev' },
  ],
  main: [
    { from:'Int1', to:'Int2' }, { from:'Int2', to:'Int1' },
    { from:'Int2', to:'Int4' }, { from:'Int4', to:'Int2' },
  ],
  pedestrian: [],   // single node, no inter-junction edges
  hexagon: [
    { from:'Int1', to:'Int2' }, { from:'Int2', to:'Int1' },
    { from:'Int2', to:'Int3' }, { from:'Int3', to:'Int2' },
    { from:'Int3', to:'Int4' }, { from:'Int4', to:'Int3' },
    { from:'Int4', to:'Int5' }, { from:'Int5', to:'Int4' },
    { from:'Int5', to:'Int6' }, { from:'Int6', to:'Int5' },
    { from:'Int6', to:'Int1' }, { from:'Int1', to:'Int6' },
    { from:'Int1', to:'L1' },   { from:'Int2', to:'L2' },
    { from:'Int3', to:'L3' },   { from:'Int4', to:'L4' },
    { from:'Int5', to:'L5' },   { from:'Int6', to:'L6' },
    { from:'L1',   to:'L2' },   { from:'L2',   to:'L1' },
    { from:'L2',   to:'L3' },   { from:'L3',   to:'L2' },
    { from:'L3',   to:'L4' },   { from:'L4',   to:'L3' },
    { from:'L4',   to:'L5' },   { from:'L5',   to:'L4' },
    { from:'L5',   to:'L6' },   { from:'L6',   to:'L5' },
    { from:'L6',   to:'L1' },   { from:'L1',   to:'L6' },
  ],
};

const NODE_RADII = { bottleneck: 18, main: 16, pedestrian: 24, hexagon: 11 };

// ── Global state ────────────────────────────────────────────────────────────
let scenario  = 'main';
let seedVal   = 42;
let data      = null;
let step      = 0;       // float; fractional part drives vehicle interpolation
let playing   = false;
let speed     = 1.0;
let stepsPerSec = 4;     // decision steps per real second at speed=1
let lastTs    = null;
let rafId     = null;
let lastIntStep = -1;
let toastTimer  = null;

// Canvas state
let ftCtx, ftW, ftH;
let rlCtx, rlW, rlH;

// Coordinate transform (shared; both panels use same network bbox)
let xform = null;   // { offsetX, offsetY, scale, flipY: fn(y) }

// Per-step vehicle maps (cached to avoid rebuilding every frame)
const vehMapCache = { ft: new Map(), rl: new Map() };  // stepIdx → Map<id, {x,y,spd}>

// ══════════════════════════════════════════════════════════════════════════
//  Bootstrap
// ══════════════════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', async () => {
  const params = new URLSearchParams(window.location.search);
  scenario = params.get('s')    || 'main';
  seedVal  = parseInt(params.get('seed') || '42', 10);

  document.getElementById('sc-select').value = scenario;
  setupSeedSelect();
  setupControls();
  setupCanvases();

  await loadData(scenario, seedVal);
  startLoop();
});

// ══════════════════════════════════════════════════════════════════════════
//  Data loading
// ══════════════════════════════════════════════════════════════════════════
async function loadData(sc, seed) {
  const fname = `data/anim_${sc}_s${seed}.json`;
  try {
    data = await fetch(fname).then(r => {
      if (!r.ok) throw new Error(r.status);
      return r.json();
    });
    scenario = sc;
    vehMapCache.ft.clear();
    vehMapCache.rl.clear();
    step = 0; lastIntStep = -1;

    buildXform(ftW || 500, ftH || 350);

    document.getElementById('step-total').textContent = data.n_steps;
    document.getElementById('scrubber').max = data.n_steps - 1;
    setupScrubberWarmup();
    renderLog();
    updateScrubber();
  } catch(e) {
    console.error('Failed to load', fname, e);
    showDataMissing(sc, seed);
  }
}

function showDataMissing(sc, seed) {
  ['ft-wrap','rl-wrap'].forEach(id => {
    const w = document.getElementById(id);
    if (!w) return;
    const c = w.querySelector('canvas');
    if (!c) return;
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.fillStyle = '#060b14';
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.fillStyle = 'rgba(239,68,68,.8)';
    ctx.font = 'bold 13px Inter,sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    const msg = location.protocol === 'file:'
      ? 'Opened as file:// — serve over HTTP: cd website && python -m http.server 8000'
      : `Data not found — run: python generate_anim_data.py --scenario ${sc}`;
    ctx.fillText(msg, c.width / 2, c.height / 2);
  });
}

// ══════════════════════════════════════════════════════════════════════════
//  Coordinate transform: SUMO metres → canvas pixels
//  SUMO Y increases upward; canvas Y increases downward → flip needed.
// ══════════════════════════════════════════════════════════════════════════
function buildXform(W, H) {
  if (!data?.network?.bbox) return;
  const b    = data.network.bbox;
  const bw   = b.max_x - b.min_x || 1;
  const bh   = b.max_y - b.min_y || 1;
  const pad  = 36;
  const scaleX = (W - pad * 2) / bw;
  const scaleY = (H - pad * 2) / bh;
  const scale  = Math.min(scaleX, scaleY);
  const offsetX = pad + ((W - pad * 2) - bw * scale) / 2;
  const offsetY = pad + ((H - pad * 2) - bh * scale) / 2;

  xform = {
    toCanvasX: (sx) => offsetX + (sx - b.min_x) * scale,
    toCanvasY: (sy) => H - (offsetY + (sy - b.min_y) * scale),  // flip Y
    scale,
  };
}

function nodePos(nodeId, W, H) {
  const j = data?.network?.junctions?.[nodeId];
  if (!j || !xform) return null;
  return { x: xform.toCanvasX(j.x), y: xform.toCanvasY(j.y) };
}

// ══════════════════════════════════════════════════════════════════════════
//  Canvas setup
// ══════════════════════════════════════════════════════════════════════════
function setupCanvases() {
  setupOneCanvas('canvas-ft', 'ft-wrap', true);
  setupOneCanvas('canvas-rl', 'rl-wrap', false);
}

function setupOneCanvas(cid, wid, isFt) {
  const wrap   = document.getElementById(wid);
  const canvas = document.getElementById(cid);
  const ro = new ResizeObserver(() => resizeCanvas(canvas, wrap, isFt));
  ro.observe(wrap);
  resizeCanvas(canvas, wrap, isFt);
}

function resizeCanvas(canvas, wrap, isFt) {
  const dpr = window.devicePixelRatio || 1;
  const w   = wrap.clientWidth  || 500;
  const h   = Math.max(280, Math.min(480, w * 0.68));
  canvas.style.width  = w + 'px';
  canvas.style.height = h + 'px';
  canvas.width  = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  if (isFt) { ftCtx = ctx; ftW = w; ftH = h; }
  else       { rlCtx = ctx; rlW = w; rlH = h; }

  buildXform(w, h);
}

// ══════════════════════════════════════════════════════════════════════════
//  Vehicle data helpers
// ══════════════════════════════════════════════════════════════════════════

/** Build Map<id → {x,y,spd}> for a given step index, with caching. */
function getVehMap(side, stepIdx) {
  const cache = vehMapCache[side];
  if (cache.has(stepIdx)) return cache.get(stepIdx);

  const arr = side === 'ft'
    ? data?.fixed_time?.vehicles?.[stepIdx]
    : data?.ppo_best?.vehicles?.[stepIdx];

  const map = new Map();
  if (arr) {
    for (let i = 0; i + 3 < arr.length; i += 4) {
      map.set(arr[i], { x: arr[i+1], y: arr[i+2], spd: arr[i+3] });
    }
  }

  // Keep cache size bounded (keep only steps within ±5 of current)
  if (cache.size > 12) {
    const oldest = cache.keys().next().value;
    cache.delete(oldest);
  }
  cache.set(stepIdx, map);
  return map;
}

/** Interpolated vehicle list at float step t.
    Vehicles present at both t0 and t1 → position/speed interpolated.
    Vehicles only at t0 → fade out in first half of the step.
    Vehicles only at t1 → fade in in second half.
    When paused: t is integer → frac=0 → exact captured positions. */
function getVehiclesAt(side, t) {
  const n   = data?.n_steps || 300;
  const t0  = Math.max(0, Math.min(Math.floor(t), n - 1));
  const t1  = Math.min(t0 + 1, n - 1);
  const frac = t - t0;

  const m0 = getVehMap(side, t0);
  const m1 = getVehMap(side, t1);

  const out = [];

  m0.forEach((v0, id) => {
    const v1 = m1.get(id);
    if (v1) {
      // Both steps have this vehicle — interpolate
      out.push({
        x:     v0.x + (v1.x - v0.x) * frac,
        y:     v0.y + (v1.y - v0.y) * frac,
        spd:   v0.spd + (v1.spd - v0.spd) * frac,
        alpha: 1.0,
      });
    } else if (frac < 0.5) {
      // Vehicle left the network — fade out
      out.push({ ...v0, alpha: 1 - frac * 2 });
    }
  });

  if (frac >= 0.5) {
    m1.forEach((v1, id) => {
      if (!m0.has(id)) {
        // New vehicle entered — fade in
        out.push({ ...v1, alpha: (frac - 0.5) * 2 });
      }
    });
  }

  return out;
}

// ══════════════════════════════════════════════════════════════════════════
//  Drawing
// ══════════════════════════════════════════════════════════════════════════
function drawFrame(t) {
  if (!data || !ftCtx || !rlCtx || !xform) return;

  const ftVeh = getVehiclesAt('ft', t);
  const rlVeh = getVehiclesAt('rl', t);

  const t0     = Math.floor(t);
  const warmup = data.warmup || 30;
  const isWarm = t0 < warmup;

  const ftPerTl = getPerTlAt('ft', t0);
  const rlPerTl = getPerTlAt('rl', t0);

  drawPanel(ftCtx, ftW, ftH, ftVeh, ftPerTl, 'ft', isWarm);
  drawPanel(rlCtx, rlW, rlH, rlVeh, rlPerTl, 'rl', isWarm);
  updateMetricReadouts(t0);
}

function getPerTlAt(side, t0) {
  const src = side === 'ft' ? data.fixed_time : data.ppo_best;
  const pt  = {};
  if (!src?.per_tl) return pt;
  Object.keys(src.per_tl).forEach(tl => {
    const d = src.per_tl[tl];
    pt[tl] = {
      stopped_ratio: d.stopped_ratio?.[t0] ?? 0,
      waiting_time:  d.waiting_time?.[t0]  ?? 0,
      queue_length:  d.queue_length?.[t0]  ?? 0,
      phase:         d.phase?.[t0]         ?? 0,
    };
  });
  return pt;
}

function drawPanel(ctx, W, H, vehicles, perTl, side, isWarmup) {
  ctx.clearRect(0, 0, W, H);

  // Dark background
  ctx.fillStyle = '#060b14';
  ctx.fillRect(0, 0, W, H);

  // Subtle grid
  ctx.strokeStyle = 'rgba(255,255,255,.02)';
  ctx.lineWidth = 1;
  for (let x = 0; x < W; x += 50) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }
  for (let y = 0; y < H; y += 50) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }

  if (!data?.network?.junctions || !xform) return;

  // 1 — Road edges
  const edges = EDGES[scenario] || [];
  edges.forEach(e => {
    const p1 = nodePos(e.from, W, H);
    const p2 = nodePos(e.to,   W, H);
    if (!p1 || !p2) return;
    drawRoad(ctx, p1, p2, perTl[e.to], W, H);
  });

  // 2 — Actual vehicle dots
  drawVehicles(ctx, vehicles, W, H);

  // 3 — Junction nodes (on top of vehicles)
  const nodeR = NODE_RADII[scenario] || 14;
  Object.keys(data.network.junctions).forEach(id => {
    const p  = nodePos(id, W, H);
    if (!p) return;
    const td = perTl[id] || {};
    drawNode(ctx, p.x, p.y, nodeR, id, td.stopped_ratio ?? 0, td.phase ?? 0, side);
  });

  // 4 — Warmup overlay
  if (isWarmup) {
    ctx.fillStyle = 'rgba(0,0,0,.6)';
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = 'rgba(255,255,255,.5)';
    ctx.font = 'bold 13px Inter,sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('Warm-up period — excluded from aggregate', W/2, H/2);
  }

  // 5 — Side badge
  ctx.font = 'bold 11px Inter,sans-serif';
  ctx.textAlign = 'left'; ctx.textBaseline = 'top';
  ctx.fillStyle = side === 'ft' ? 'rgba(249,115,22,.75)' : 'rgba(16,185,129,.75)';
  ctx.fillText(side === 'ft' ? '🟠 Fixed-Time' : '🟢 PPO (RL)', 10, 8);
}

// ── Road segment (SUMO data-aware: direction matters) ──────────────────────
function drawRoad(ctx, p1, p2, dstData, W, H) {
  const dx  = p2.x - p1.x, dy = p2.y - p1.y;
  const len = Math.hypot(dx, dy);
  if (len < 2) return;
  // Offset perpendicular so bidirectional roads don't overlap
  const ox = (-dy / len) * 2.5, oy = (dx / len) * 2.5;

  // Road base
  ctx.strokeStyle = 'rgba(25,40,65,.95)';
  ctx.lineWidth = 8; ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(p1.x + ox, p1.y + oy);
  ctx.lineTo(p2.x + ox, p2.y + oy);
  ctx.stroke();

  // Congestion heat — fills from destination node backward
  const ratio = dstData?.stopped_ratio ?? 0;
  if (ratio > 0.05) {
    const frac = Math.min(0.88, ratio * 1.15);
    const ex = p2.x + ox + (p1.x - p2.x) * frac;
    const ey = p2.y + oy + (p1.y - p2.y) * frac;
    const grd = ctx.createLinearGradient(p2.x + ox, p2.y + oy, ex, ey);
    grd.addColorStop(0,   congestionColor(ratio, 0.78));
    grd.addColorStop(0.6, congestionColor(ratio, 0.28));
    grd.addColorStop(1,   'rgba(0,0,0,0)');
    ctx.strokeStyle = grd;
    ctx.lineWidth   = 5; ctx.lineCap = 'butt';
    ctx.beginPath();
    ctx.moveTo(p2.x + ox, p2.y + oy);
    ctx.lineTo(ex, ey);
    ctx.stroke();
    ctx.lineCap = 'round';
  }

  // Lane marking
  ctx.strokeStyle = 'rgba(255,255,255,.05)';
  ctx.lineWidth = 0.7;
  ctx.setLineDash([5, 7]);
  ctx.beginPath();
  ctx.moveTo(p1.x + ox, p1.y + oy);
  ctx.lineTo(p2.x + ox, p2.y + oy);
  ctx.stroke();
  ctx.setLineDash([]);
}

// ── Vehicle dots from actual SUMO data ─────────────────────────────────────
function drawVehicles(ctx, vehicles, W, H) {
  if (!xform || !vehicles.length) return;

  vehicles.forEach(v => {
    const cx = xform.toCanvasX(v.x);
    const cy = xform.toCanvasY(v.y);

    // Clip to canvas with a small margin
    if (cx < -10 || cx > W + 10 || cy < -10 || cy > H + 10) return;

    // Colour: red (0 km/h) → amber (20 km/h) → green (50+ km/h)
    const t  = Math.min(1, v.spd / 50);
    const r  = Math.round(239 - (239 - 16)  * t);
    const g  = Math.round(68  + (185 - 68)  * t);
    const b  = Math.round(68  + (129 - 68)  * t);

    ctx.globalAlpha = (v.alpha ?? 1) * 0.88;
    ctx.fillStyle   = `rgb(${r},${g},${b})`;

    // Stopped vehicles: slightly larger red square
    if (v.spd < 1) {
      ctx.fillRect(cx - 2.5, cy - 2.5, 5, 5);
    } else {
      // Moving vehicles: small oriented circle
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  });
  ctx.globalAlpha = 1;
}

// ── Junction node ──────────────────────────────────────────────────────────
function drawNode(ctx, x, y, r, id, ratio, phase, side) {
  const col = congestionColor(ratio);

  // Glow for congested nodes
  if (ratio > 0.2) {
    const gr = r + 6 + ratio * 8;
    const grd = ctx.createRadialGradient(x, y, r * 0.6, x, y, gr);
    grd.addColorStop(0, congestionColor(ratio, ratio * 0.45));
    grd.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = grd;
    ctx.beginPath(); ctx.arc(x, y, gr, 0, Math.PI * 2); ctx.fill();
  }

  // Fill
  ctx.fillStyle   = col;
  ctx.strokeStyle = 'rgba(255,255,255,.28)';
  ctx.lineWidth   = 1.5;
  ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill(); ctx.stroke();

  // Phase arc on RL panel — shows which green phase the agent selected
  if (side === 'rl') {
    const phaseColors = ['#10b981','#06b6d4','#f59e0b','#a78bfa','#f43f5e','#84cc16'];
    const start = (phase / 4) * Math.PI * 2 - Math.PI / 2;
    ctx.strokeStyle = phaseColors[phase % phaseColors.length];
    ctx.lineWidth   = 3;
    ctx.beginPath();
    ctx.arc(x, y, r + 4, start, start + Math.PI * 0.5);
    ctx.stroke();
  }

  // Label
  const shortId = id.replace('TL_', '').slice(0, 5);
  ctx.fillStyle    = 'rgba(255,255,255,.9)';
  ctx.font         = `bold ${Math.max(7, r - 5)}px Inter,sans-serif`;
  ctx.textAlign    = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(shortId, x, y);
}

// ── Congestion → RGB ───────────────────────────────────────────────────────
function congestionColor(ratio, alpha = 1) {
  let r, g, b;
  if (ratio < 0.5) {
    const t = ratio * 2;
    r = Math.round(16  + (245 - 16)  * t);
    g = Math.round(185 + (158 - 185) * t);
    b = Math.round(129 + (11  - 129) * t);
  } else {
    const t = (ratio - 0.5) * 2;
    r = Math.round(245 + (239 - 245) * t);
    g = Math.round(158 + (68  - 158) * t);
    b = Math.round(11  + (68  - 11)  * t);
  }
  return alpha < 1 ? `rgba(${r},${g},${b},${alpha})` : `rgb(${r},${g},${b})`;
}

// ── Metric readouts ────────────────────────────────────────────────────────
function updateMetricReadouts(t0) {
  const idx   = Math.max(0, t0 - (data.warmup || 30));
  const ft    = data.fixed_time;
  const rl    = data.ppo_best;
  const FMT   = [
    { key:'waiting_time',  label:'Wait',  unit:'s',    p:1 },
    { key:'queue_length',  label:'Queue', unit:'veh',  p:1 },
    { key:'stopped_ratio', label:'Stop',  unit:'',     p:3 },
    { key:'throughput',    label:'Thr',   unit:'/st',  p:0 },
  ];

  [['ft-metrics', ft], ['rl-metrics', rl]].forEach(([id, src]) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.innerHTML = FMT.map(m => {
      const arr = src?.[m.key];
      const v   = arr && idx >= 0 && idx < arr.length ? arr[idx] : 0;
      return `<span class="ph-m">${m.label}: <span class="v">${v.toFixed(m.p)}${m.unit}</span></span>`;
    }).join('');
  });

  // Vehicles count
  const nFt = Math.floor((data.fixed_time?.vehicles?.[t0]?.length || 0) / 4);
  const nRl = Math.floor((data.ppo_best?.vehicles?.[t0]?.length  || 0) / 4);
  document.getElementById('step-num').textContent = t0;
  document.getElementById('sim-time').textContent  = `${t0 * (data.delta_time || 10)}s`;

  const ftVEl = document.getElementById('ft-veh-count');
  const rlVEl = document.getElementById('rl-veh-count');
  if (ftVEl) ftVEl.textContent = `${nFt} veh`;
  if (rlVEl) rlVEl.textContent = `${nRl} veh`;
}

// ══════════════════════════════════════════════════════════════════════════
//  Highlight system
// ══════════════════════════════════════════════════════════════════════════
function checkHighlights(iStep) {
  const hl = data?.highlights?.find(h => h.step === iStep);
  if (hl) showToast(hl);
}

function showToast(hl) {
  clearTimeout(toastTimer);
  const toast = document.getElementById('hl-toast');
  const simEl = document.getElementById('ht-simtime');
  const rows  = document.getElementById('ht-rows');
  const bar   = document.getElementById('ht-bar');

  simEl.textContent = `${hl.sim_seconds}s simulated  (step ${hl.step})`;

  const ftQpct = hl.ft_queue > 0
    ? ((hl.ft_queue - hl.rl_queue) / hl.ft_queue * 100).toFixed(1)
    : '0.0';

  rows.innerHTML = `
    <div class="ht-row">
      <span class="ht-metric">Waiting Time</span>
      <span class="ht-ft">${hl.ft_wait.toFixed(1)}s</span>
      <span class="ht-sep">→</span>
      <span class="ht-rl">${hl.rl_wait.toFixed(1)}s</span>
      <span class="ht-pct pos">▼ ${hl.wait_pct.toFixed(1)}%</span>
    </div>
    <div class="ht-row">
      <span class="ht-metric">Queue Length</span>
      <span class="ht-ft">${hl.ft_queue.toFixed(1)} veh</span>
      <span class="ht-sep">→</span>
      <span class="ht-rl">${hl.rl_queue.toFixed(1)} veh</span>
      <span class="ht-pct pos">▼ ${ftQpct}%</span>
    </div>
    <div class="ht-row">
      <span class="ht-metric">Vehicles on Net</span>
      <span class="ht-ft">${hl.ft_vehicles ?? '—'}</span>
      <span class="ht-sep">vs</span>
      <span class="ht-rl">${hl.rl_vehicles ?? '—'}</span>
      <span class="ht-pct" style="color:var(--txt-2)">count</span>
    </div>
  `;
  bar.style.width = Math.min(100, hl.wait_pct) + '%';
  toast.classList.add('show');
  toastTimer = setTimeout(() => toast.classList.remove('show'), 4500);
}

function renderLog() {
  const list  = document.getElementById('log-list');
  const count = document.getElementById('log-count');
  if (!list) return;

  const hls = data?.highlights || [];
  if (count) count.textContent = hls.length ? `${hls.length} moments` : '';

  if (!hls.length) {
    list.innerHTML = `<div class="log-empty">No significant improvements for this seed. Try a different seed.</div>`;
    return;
  }

  list.innerHTML = hls.map(h => `
    <div class="log-item" data-step="${h.step}">
      <span class="log-step">Step ${h.step}</span>
      <span class="log-badge ${h.wait_pct >= 70 ? '' : 'minor'}">▼ ${h.wait_pct.toFixed(1)}%</span>
      <span class="log-desc">
        Wait ${h.ft_wait.toFixed(1)}s → ${h.rl_wait.toFixed(1)}s
        &nbsp;·&nbsp;
        Queue ${h.ft_queue.toFixed(1)} → ${h.rl_queue.toFixed(1)} veh
        &nbsp;·&nbsp;
        FT has ${h.ft_vehicles ?? '?'} vehicles on network, RL has ${h.rl_vehicles ?? '?'}
      </span>
      <span class="log-time">${h.sim_seconds}s</span>
    </div>
  `).join('');

  list.querySelectorAll('.log-item').forEach(el => {
    el.addEventListener('click', () => {
      const s = parseInt(el.dataset.step, 10);
      step = s; updateScrubber();
      drawFrame(step);
      showToast(data.highlights.find(h => h.step === s));
      list.querySelectorAll('.log-item').forEach(i => i.classList.remove('current'));
      el.classList.add('current');
    });
  });
}

// ══════════════════════════════════════════════════════════════════════════
//  Controls
// ══════════════════════════════════════════════════════════════════════════
function setupControls() {
  document.getElementById('sc-select').addEventListener('change', async e => {
    scenario = e.target.value;
    setupSeedSelect();
    await loadData(scenario, seedVal);
  });
  document.getElementById('seed-select').addEventListener('change', async e => {
    seedVal = parseInt(e.target.value, 10);
    await loadData(scenario, seedVal);
  });

  document.getElementById('btn-play').addEventListener('click', togglePlay);
  document.getElementById('btn-prev').addEventListener('click', () => {
    step = Math.max(0, Math.floor(step) - 10);
    updateScrubber(); drawFrame(step);
  });
  document.getElementById('btn-next').addEventListener('click', () => {
    step = Math.min((data?.n_steps || 300) - 1, Math.floor(step) + 10);
    updateScrubber(); drawFrame(step);
  });

  document.querySelectorAll('[data-speed]').forEach(btn => {
    btn.addEventListener('click', () => {
      speed = parseFloat(btn.dataset.speed);
      document.querySelectorAll('[data-speed]').forEach(b => b.classList.toggle('active', b === btn));
    });
  });

  const scrubber = document.getElementById('scrubber');
  scrubber.addEventListener('input', () => {
    step = parseInt(scrubber.value, 10);
    updateScrubber();
    drawFrame(step);   // redraw immediately on drag — vehicles freeze at dragged step
  });
}

function setupSeedSelect() {
  const sel   = document.getElementById('seed-select');
  sel.innerHTML = '';
  const seeds = {
    bottleneck: [42,123,456,789,1000],
    main:       [42,123,456,789,1000,2000,3000,4000,5000,6000],
    pedestrian: [42,123,456,789,1000],
    hexagon:    [42,123,456,789,1000],
  };
  (seeds[scenario] || [42]).forEach(s => {
    const opt = document.createElement('option');
    opt.value = s; opt.textContent = `Seed ${s}`;
    if (s === seedVal) opt.selected = true;
    sel.appendChild(opt);
  });
}

function togglePlay() {
  playing = !playing;
  document.getElementById('btn-play').textContent = playing ? '⏸' : '▶';
  if (playing && data && step >= (data.n_steps - 1)) step = 0;
  if (playing) lastTs = null;
}

function setupScrubberWarmup() {
  const warmup = data?.warmup || 30;
  const total  = data?.n_steps || 300;
  const mark   = document.getElementById('warmup-mark');
  if (mark) mark.style.left = (warmup / total * 100) + '%';
  const lbl = document.getElementById('warmup-label');
  if (lbl) lbl.textContent = `▲ step ${warmup}`;
}

function updateScrubber() {
  if (!data) return;
  const t = Math.floor(step);
  document.getElementById('scrubber').value = t;
  document.getElementById('scrubber-fill').style.width = (t / (data.n_steps - 1) * 100) + '%';
  document.getElementById('step-num').textContent = t;
  document.getElementById('sim-time').textContent  = `${t * (data.delta_time || 10)}s`;
  document.getElementById('scrubber-time').textContent =
    `Step ${t} of ${data.n_steps} · ${t * (data.delta_time||10)}s simulated`;
}

// ══════════════════════════════════════════════════════════════════════════
//  Animation loop
//  Key invariant: step only advances when playing=true.
//  drawFrame is called every RAF tick (needed for real-time canvas refresh),
//  but vehicles are drawn from data at Math.floor(step) with fractional
//  interpolation only during playback — zero motion when paused.
// ══════════════════════════════════════════════════════════════════════════
function startLoop() {
  if (rafId) cancelAnimationFrame(rafId);
  lastTs = null;
  rafId  = requestAnimationFrame(loop);
}

function loop(ts) {
  rafId = requestAnimationFrame(loop);
  const dt = (lastTs && playing) ? Math.min((ts - lastTs) / 1000, 0.1) : 0;
  lastTs   = ts;

  if (!data) return;

  if (playing) {
    step = step + speed * stepsPerSec * dt;
    const n = data.n_steps;
    if (step >= n - 1) {
      step = n - 1; playing = false;
      document.getElementById('btn-play').textContent = '▶';
    }

    const iStep = Math.floor(step);
    if (iStep !== lastIntStep) {
      lastIntStep = iStep;
      checkHighlights(iStep);

      // Highlight current log item
      document.querySelectorAll('.log-item').forEach(el => {
        const s = parseInt(el.dataset.step, 10);
        el.classList.toggle('current', s === iStep && data.highlights?.some(h => h.step === iStep));
      });
    }
    updateScrubber();
  }

  drawFrame(step);
}
