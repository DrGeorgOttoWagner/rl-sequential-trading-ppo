/* Dependency-free canvas/SVG charts (zoom, pan, hover, step series, markers, progressive reveal). */
'use strict';

class LineChart {
  /**
   * opts: { yFormat, yDomain: [min,max] | null, area: bool, zeroLine: bool, stepY: [{v,label}], yTicks: n,
   *         categorical: {0:'CASH',1:'INVESTED'} | null, legend: bool }
   */
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.opts = Object.assign({ yFormat: v => v.toFixed(2), yDomain: null, area: false, zeroLine: false, categorical: null, legend: true, yTicks: 5, padY: 0.06 }, opts);
    this.series = [];
    this.markers = [];
    this.N = 0;
    this.visible = Infinity;
    this.view = [0, 1];       // x-domain in index space
    this.hover = null;
    this.pad = { l: 64, r: 16, t: 14, b: 30 };
    this.tooltip = el('div', { class: 'chart-tooltip' });
    canvas.parentElement.style.position = 'relative';
    canvas.parentElement.appendChild(this.tooltip);
    this._bind();
    this._ro = new ResizeObserver(() => this.render());
    this._ro.observe(canvas.parentElement);
  }

  setSeries(series, dates) {
    this.series = series; this.dates = dates || (series[0] && series[0].x) || [];
    this.N = Math.max(0, ...series.map(s => s.y.length));
    this.resetView();
  }
  setMarkers(m) { this.markers = m || []; this.render(); }
  setVisible(n) { this.visible = n; this.render(); }
  resetView() { this.view = [0, Math.max(1, this.N - 1)]; this.render(); }

  _bind() {
    const c = this.canvas;
    let drag = null;
    c.addEventListener('wheel', ev => {
      ev.preventDefault();
      const { l } = this.pad, w = this._plotW();
      const px = (ev.offsetX - l) / w;
      const [a, b] = this.view, span = b - a;
      const f = ev.deltaY < 0 ? 0.8 : 1.25;
      let ns = Math.max(10, Math.min(this.N - 1, span * f));
      let na = a + (span - ns) * px;
      na = Math.max(0, Math.min(this.N - 1 - ns, na));
      this.view = [na, na + ns]; this.render();
    }, { passive: false });
    c.addEventListener('mousedown', ev => { drag = { x: ev.offsetX, view: [...this.view] }; c.style.cursor = 'grabbing'; });
    window.addEventListener('mouseup', () => { drag = null; c.style.cursor = 'crosshair'; });
    c.addEventListener('mousemove', ev => {
      if (drag) {
        const w = this._plotW(), [a, b] = drag.view, span = b - a;
        let na = a - (ev.offsetX - drag.x) / w * span;
        na = Math.max(0, Math.min(this.N - 1 - span, na));
        this.view = [na, na + span];
      }
      this.hover = this._idxAt(ev.offsetX);
      this.render(); this._tooltip(ev);
    });
    c.addEventListener('mouseleave', () => { this.hover = null; this.tooltip.style.display = 'none'; this.render(); });
    c.addEventListener('dblclick', () => this.resetView());
    c.style.cursor = 'crosshair';
  }

  _plotW() { return this.canvas.clientWidth - this.pad.l - this.pad.r; }
  _plotH() { return this.canvas.clientHeight - this.pad.t - this.pad.b; }
  _xToPx(i) { const [a, b] = this.view; return this.pad.l + (i - a) / (b - a) * this._plotW(); }
  _yToPx(v) { const [lo, hi] = this.yDom; return this.pad.t + (1 - (v - lo) / (hi - lo)) * this._plotH(); }
  _idxAt(px) {
    const [a, b] = this.view;
    const i = Math.round(a + (px - this.pad.l) / this._plotW() * (b - a));
    const lim = Math.min(this.N, this.visible) - 1;
    return Math.max(0, Math.min(lim, i));
  }

  _computeYDomain() {
    if (this.opts.yDomain) { this.yDom = this.opts.yDomain; return; }
    const [a, b] = this.view, lim = Math.min(this.N, this.visible);
    let lo = Infinity, hi = -Infinity;
    for (const s of this.series) {
      if (s.hidden) continue;
      for (let i = Math.max(0, Math.floor(a)); i <= Math.min(lim - 1, Math.ceil(b)); i++) {
        const v = s.y[i]; if (v === undefined || v === null) continue;
        if (v < lo) lo = v; if (v > hi) hi = v;
      }
    }
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    if (this.opts.zeroLine) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
    if (hi - lo < 1e-9) { hi = lo + 1; }
    const p = (hi - lo) * this.opts.padY;
    this.yDom = [lo - p, hi + p];
  }

  render() {
    const c = this.canvas, dpr = window.devicePixelRatio || 1;
    const W = c.clientWidth, H = c.clientHeight;
    if (W === 0 || H === 0) return;
    if (c.width !== W * dpr || c.height !== H * dpr) { c.width = W * dpr; c.height = H * dpr; }
    const ctx = this.ctx; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    if (!this.series.length || !this.N) return;
    this._computeYDomain();
    const { l, t } = this.pad, pw = this._plotW(), ph = this._plotH();
    const [a, b] = this.view, lim = Math.min(this.N, this.visible);
    ctx.font = '11px "IBM Plex Sans", system-ui, sans-serif';

    // Y grid + labels
    ctx.strokeStyle = COLORS.grid; ctx.fillStyle = COLORS.muted; ctx.lineWidth = 1; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    const ticks = this.opts.categorical ? Object.keys(this.opts.categorical).map(Number) : niceTicks(this.yDom[0], this.yDom[1], this.opts.yTicks);
    for (const v of ticks) {
      if (v < this.yDom[0] || v > this.yDom[1]) continue;
      const y = this._yToPx(v);
      ctx.beginPath(); ctx.moveTo(l, y); ctx.lineTo(l + pw, y); ctx.stroke();
      ctx.fillText(this.opts.categorical ? this.opts.categorical[v] : this.opts.yFormat(v), l - 8, y);
    }
    if (this.opts.zeroLine && !this.opts.categorical) {
      const y = this._yToPx(0); ctx.strokeStyle = '#4b5563'; ctx.beginPath(); ctx.moveTo(l, y); ctx.lineTo(l + pw, y); ctx.stroke();
    }
    // X labels
    ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = COLORS.muted;
    const nx = Math.max(2, Math.floor(pw / 110));
    for (let k = 0; k <= nx; k++) {
      const i = Math.round(a + (b - a) * k / nx);
      if (i < 0 || i >= this.N) continue;
      const x = this._xToPx(i);
      ctx.strokeStyle = COLORS.grid; ctx.beginPath(); ctx.moveTo(x, t); ctx.lineTo(x, t + ph); ctx.stroke();
      ctx.fillText(this.dates[i] || String(i), x, t + ph + 6);
    }
    // Series
    ctx.save(); ctx.beginPath(); ctx.rect(l, t, pw, ph); ctx.clip();
    const i0 = Math.max(0, Math.floor(a)), i1 = Math.min(lim - 1, Math.ceil(b));
    for (const s of this.series) {
      if (s.hidden) continue;
      ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.6; ctx.setLineDash(s.dash || []);
      ctx.beginPath();
      let started = false, lastY = null;
      for (let i = i0; i <= i1; i++) {
        const v = s.y[i]; if (v === undefined || v === null) continue;
        const x = this._xToPx(i), y = this._yToPx(v);
        if (!started) { ctx.moveTo(x, y); started = true; }
        else if (s.step) { ctx.lineTo(x, lastY); ctx.lineTo(x, y); }
        else ctx.lineTo(x, y);
        lastY = y;
      }
      ctx.stroke(); ctx.setLineDash([]);
      if (s.area && started) {
        const base = this._yToPx(s.areaBase !== undefined ? s.areaBase : this.yDom[0]);
        ctx.lineTo(this._xToPx(i1), base); ctx.lineTo(this._xToPx(i0), base); ctx.closePath();
        ctx.fillStyle = s.color + '33'; ctx.fill();
      }
    }
    // Markers (BUY / SELL)
    for (const m of this.markers) {
      if (m.i < i0 || m.i > i1) continue;
      const x = this._xToPx(m.i), y = this._yToPx(m.y);
      ctx.fillStyle = m.type === 'BUY' ? COLORS.buy : COLORS.sell;
      ctx.beginPath();
      if (m.type === 'BUY') { ctx.moveTo(x, y - 6); ctx.lineTo(x - 4, y + 1); ctx.lineTo(x + 4, y + 1); }
      else { ctx.moveTo(x, y + 6); ctx.lineTo(x - 4, y - 1); ctx.lineTo(x + 4, y - 1); }
      ctx.closePath(); ctx.fill();
    }
    // Replay cursor / hover line
    const cursor = this.hover !== null ? this.hover : (this.visible < this.N ? this.visible - 1 : null);
    if (cursor !== null && cursor >= i0 && cursor <= i1) {
      const x = this._xToPx(cursor);
      ctx.strokeStyle = '#cbd5e1'; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(x, t); ctx.lineTo(x, t + ph); ctx.stroke(); ctx.setLineDash([]);
      for (const s of this.series) {
        if (s.hidden) continue; const v = s.y[cursor]; if (v === undefined || v === null) continue;
        ctx.fillStyle = s.color; ctx.beginPath(); ctx.arc(x, this._yToPx(v), 3.2, 0, Math.PI * 2); ctx.fill();
      }
    }
    ctx.restore();
    // Legend
    if (this.opts.legend) {
      let x = l + 8; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      for (const s of this.series) {
        if (s.hidden) continue;
        ctx.fillStyle = s.color; ctx.fillRect(x, t + 6, 14, 3); x += 18;
        ctx.fillStyle = COLORS.text; ctx.fillText(s.name, x, t + 7); x += ctx.measureText(s.name).width + 16;
      }
    }
  }

  _tooltip(ev) {
    const i = this.hover; if (i === null) return;
    const rows = this.series.filter(s => !s.hidden && s.y[i] !== undefined && s.y[i] !== null)
      .map(s => `<div><span class="sw" style="background:${s.color}"></span>${s.name}: <b>${(s.format || this.opts.yFormat)(s.y[i])}</b></div>`);
    const extra = this.opts.hoverExtra ? this.opts.hoverExtra(i) : '';
    this.tooltip.innerHTML = `<div class="tt-date">${this.dates[i] || i}</div>${rows.join('')}${extra}`;
    this.tooltip.style.display = 'block';
    const box = this.canvas.parentElement.getBoundingClientRect();
    let x = ev.clientX - box.left + 14, y = ev.clientY - box.top + 10;
    if (x + this.tooltip.offsetWidth > box.width - 8) x = ev.clientX - box.left - this.tooltip.offsetWidth - 14;
    this.tooltip.style.left = x + 'px'; this.tooltip.style.top = y + 'px';
  }
}

function niceTicks(lo, hi, n) {
  const span = hi - lo, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag; const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toFixed(10));
  return out;
}

/* ---------- SVG helpers ---------- */
const SVG_NS = 'http://www.w3.org/2000/svg';
function svg(tag, attrs = {}, children = []) {
  const e = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  for (const c of [].concat(children)) if (c) e.append(c);
  return e;
}
function svgText(x, y, text, attrs = {}) { const t = svg('text', Object.assign({ x, y, fill: COLORS.text, 'font-size': 11 }, attrs)); t.textContent = text; return t; }

/** Horizontal bar chart. items: [{label, value, color, note}] ; opts: {format, zero:true, width, rowH} */
function barChart(container, items, opts = {}) {
  const width = opts.width || container.clientWidth || 420, rowH = opts.rowH || 22, labelW = opts.labelW || 110, valW = 90;
  const h = items.length * rowH + 10, pw = width - labelW - valW - 20;
  const vals = items.map(i => i.value).filter(v => v !== null && v !== undefined);
  let lo = Math.min(0, ...vals), hi = Math.max(0, ...vals); if (hi - lo < 1e-9) hi = lo + 1;
  const x = v => labelW + (v - lo) / (hi - lo) * pw;
  const s = svg('svg', { width, height: h, viewBox: `0 0 ${width} ${h}`, class: 'barchart' });
  s.append(svg('line', { x1: x(0), x2: x(0), y1: 4, y2: h - 4, stroke: '#4b5563' }));
  items.forEach((it, k) => {
    const y = 6 + k * rowH;
    s.append(svgText(labelW - 8, y + rowH / 2 + 4, it.label, { 'text-anchor': 'end', fill: it.dim ? COLORS.muted : COLORS.text }));
    if (it.value === null || it.value === undefined) { s.append(svgText(x(0) + 6, y + rowH / 2 + 4, it.note || 'undefined', { fill: COLORS.muted })); return; }
    const x0 = Math.min(x(0), x(it.value)), w = Math.abs(x(it.value) - x(0));
    s.append(svg('rect', { x: x0, y: y + 4, width: Math.max(w, 1), height: rowH - 8, fill: it.color, rx: 2, opacity: it.dim ? 0.55 : 0.95 }));
    s.append(svgText(labelW + pw + 8, y + rowH / 2 + 4, (opts.format || (v => v.toFixed(3)))(it.value), { fill: COLORS.text }));
  });
  container.replaceChildren(s);
}

/** Dot + range plot per experiment. groups: [{label, color, points:[{seed,value}], mean}] */
function dotRange(container, groups, opts = {}) {
  const width = opts.width || container.clientWidth || 520, rowH = 34, labelW = 90, h = groups.length * rowH + 30;
  const pw = width - labelW - 30;
  const all = groups.flatMap(g => g.points.map(p => p.value)).concat(opts.extra ? opts.extra.map(e => e.value) : []);
  let lo = Math.min(0, ...all), hi = Math.max(0, ...all); const pad = (hi - lo) * 0.06; lo -= pad; hi += pad;
  const x = v => labelW + (v - lo) / (hi - lo) * pw;
  const s = svg('svg', { width, height: h, viewBox: `0 0 ${width} ${h}`, class: 'dotrange' });
  for (const t of niceTicks(lo, hi, 6)) {
    s.append(svg('line', { x1: x(t), x2: x(t), y1: 6, y2: h - 22, stroke: COLORS.grid }));
    s.append(svgText(x(t), h - 8, (opts.format || fmtSigned)(t, 2), { 'text-anchor': 'middle', fill: COLORS.muted, 'font-size': 10 }));
  }
  s.append(svg('line', { x1: x(0), x2: x(0), y1: 6, y2: h - 22, stroke: '#6b7280' }));
  (opts.extra || []).forEach(e => {
    s.append(svg('line', { x1: x(e.value), x2: x(e.value), y1: 6, y2: h - 22, stroke: e.color, 'stroke-dasharray': '4 3' }));
    s.append(svgText(x(e.value) + 4, 14, e.label, { fill: e.color, 'font-size': 10 }));
  });
  groups.forEach((g, k) => {
    const y = 20 + k * rowH + rowH / 2;
    s.append(svgText(labelW - 10, y + 4, g.label, { 'text-anchor': 'end', fill: g.color, 'font-weight': 600 }));
    const vs = g.points.map(p => p.value);
    s.append(svg('line', { x1: x(Math.min(...vs)), x2: x(Math.max(...vs)), y1: y, y2: y, stroke: g.color, 'stroke-width': 2, opacity: 0.6 }));
    g.points.forEach(p => {
      const c = svg('circle', { cx: x(p.value), cy: y, r: 5, fill: g.color, stroke: '#0f1419', 'stroke-width': 1 });
      c.append(svg('title', {}, [document.createTextNode(`seed ${p.seed}: ${fmtSigned(p.value, 4)}`)]));
      s.append(c);
    });
    if (g.mean !== undefined) s.append(svg('rect', { x: x(g.mean) - 1.5, y: y - 9, width: 3, height: 18, fill: '#fff', opacity: 0.85 }));
  });
  container.replaceChildren(s);
}

/** Slope chart VALIDATION -> TEST. rows: [{label,color,a,b}] */
function slopeChart(container, rows, opts = {}) {
  const width = opts.width || container.clientWidth || 520, h = opts.height || 300, padL = 120, padR = 120, top = 30, bottom = 30;
  const all = rows.flatMap(r => [r.a, r.b]); let lo = Math.min(0, ...all), hi = Math.max(...all); const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  const y = v => top + (1 - (v - lo) / (hi - lo)) * (h - top - bottom);
  const xa = padL, xb = width - padR;
  const s = svg('svg', { width, height: h, viewBox: `0 0 ${width} ${h}`, class: 'slope' });
  s.append(svgText(xa, 16, opts.labelA || 'VALIDATION', { 'text-anchor': 'middle', fill: COLORS.muted, 'font-weight': 600 }));
  s.append(svgText(xb, 16, opts.labelB || 'TEST', { 'text-anchor': 'middle', fill: COLORS.muted, 'font-weight': 600 }));
  s.append(svg('line', { x1: xa, x2: xb, y1: y(0), y2: y(0), stroke: '#4b5563', 'stroke-dasharray': '3 3' }));
  s.append(svg('line', { x1: xa, x2: xa, y1: top, y2: h - bottom, stroke: COLORS.grid }));
  s.append(svg('line', { x1: xb, x2: xb, y1: top, y2: h - bottom, stroke: COLORS.grid }));
  const fmt = opts.format || (v => fmtSigned(v, 4));
  // Collision-avoid labels per side.
  const place = (vals) => { const ys = vals.map(v => y(v)); const order = ys.map((v, i) => i).sort((i, j) => ys[i] - ys[j]); let last = -Infinity; const out = []; for (const i of order) { let yy = Math.max(ys[i], last + 14); out[i] = yy; last = yy; } return out; };
  const la = place(rows.map(r => r.a)), lb = place(rows.map(r => r.b));
  rows.forEach((r, k) => {
    s.append(svg('line', { x1: xa, y1: y(r.a), x2: xb, y2: y(r.b), stroke: r.color, 'stroke-width': 2.5, opacity: 0.9 }));
    s.append(svg('circle', { cx: xa, cy: y(r.a), r: 5, fill: r.color }));
    s.append(svg('circle', { cx: xb, cy: y(r.b), r: 5, fill: r.color }));
    s.append(svgText(xa - 10, la[k] + 4, `${r.label} ${fmt(r.a)}`, { 'text-anchor': 'end', fill: r.color }));
    s.append(svgText(xb + 10, lb[k] + 4, `${fmt(r.b)} ${r.label}`, { fill: r.color }));
  });
  container.replaceChildren(s);
}
