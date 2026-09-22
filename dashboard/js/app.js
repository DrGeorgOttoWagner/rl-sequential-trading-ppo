/* Application wiring for the read-only evidence dashboard.
   Boot: fetch the entry point → verify (contract, binding constant, dashboard_set_id, projections)
   → fetch and hash every INCLUDED payload → render. Any failure shows an integrity error and no result.
   While no evidence package is installed, only the explanatory sections are visible. */
'use strict';

const EXPERIMENTS = ['E1', 'E2', 'E3', 'E4'];
const SPLITS = ['VALIDATION', 'TEST'];
const COLORS = { E1: '#7aa2f7', E2: '#4fc38a', E3: '#c678dd', E4: '#f2a65a' };
const PAYLOADS = {};       // logical_payload_id → parsed payload (INCLUDED only)
const PROJECTIONS = {};    // logical_payload_id → descriptor projection

/* ---------- states ---------- */
function setPanelLabels() {
  document.querySelectorAll('[data-panel-label]').forEach(b => {
    b.textContent = b.dataset.panelLabel === 'frozen' ? 'Frozen evidence' : 'Explanatory presentation';
    b.classList.add(b.dataset.panelLabel === 'frozen' ? 'badge-frozen' : 'badge-ok');
  });
}
function hideResultSections() {
  document.querySelectorAll('section[data-requires]').forEach(s => { s.hidden = true; });
  document.getElementById('claims').hidden = true;
  document.getElementById('limitations').hidden = true;
}
function renderNoPackage(reason) {
  hideResultSections();
  const box = document.getElementById('no-package'); box.hidden = false;
  document.getElementById('package-badge').textContent = 'NO EVIDENCE PACKAGE INSTALLED';
  const prov = document.getElementById('prov-table');
  prov.replaceChildren(el('tr', {}, [el('td', { text: 'Evidence package' }), el('td', { class: 'mono', text: 'none installed (' + reason + ')' })]));
  document.getElementById('notices-list').replaceChildren(el('li', { text: 'No payload was loaded and nothing was verified; this page contains no embedded, cached or fallback data.' }));
}
function renderIntegrityError(err) {
  hideResultSections();
  document.getElementById('no-package').hidden = true;
  const box = document.getElementById('load-error'); box.hidden = false;
  box.textContent = 'Integrity error at consumer step ' + (err.step || '?') + ' (' + (err.code || 'V-INPUT') + '): ' + err.message + ' — no scientific value, chart or table is shown.';
  document.getElementById('package-badge').textContent = 'EVIDENCE PACKAGE FAILED VERIFICATION';
}
function withheldNotice(text) { return el('div', { class: 'notice', text: text }); }

/* ---------- renderers (schemas follow the draft transformation register; re-reviewed with it) ---------- */
function req(obj, key, where) { if (obj === null || typeof obj !== 'object' || !(key in obj)) throw new IntegrityError(6, 'V-SCHEMA', where + ' lacks ' + key); return obj[key]; }
function isPS(x) { return x && typeof x === 'object' && (x.status === 'INCLUDED' || x.status === 'WITHHELD'); }

function renderClaims(ec11, notices) {
  const title = req(ec11, 'study_title', 'claims payload');
  document.getElementById('study-title').textContent = title;
  const lims = req(ec11, 'limitations', 'claims payload');
  document.getElementById('limitations-list').replaceChildren(...lims.map(l => el('li', { text: req(l, 'text', 'limitation') })));
  document.getElementById('limitations').hidden = false;
  const claims = req(ec11, 'claims', 'claims payload');
  document.getElementById('claims-box').replaceChildren(...claims.map(c => el('div', { class: 'card' }, [
    el('h3', {}, [el('span', { class: 'tag', text: req(c, 'claim_id', 'claim') }), document.createTextNode(' '), el('span', { class: 'badge', text: req(c, 'strength', 'claim') })]),
    el('p', { text: req(c, 'statement', 'claim') }),
    el('p', { class: 'footnote', text: 'Evidence payloads: ' + (Array.isArray(c.evidence_payloads) ? c.evidence_payloads.join(', ') : '—') }),
  ])));
  document.getElementById('claims').hidden = false;
  const nl = document.getElementById('notices-list');
  nl.replaceChildren(...(Array.isArray(notices) ? notices : []).map(n => el('li', { text: String(n) })));
}

function metricColumns(row) {
  const skip = new Set(['experiment', 'definition', 'n_seeds', 'split', 'observation', 'reward']);
  return Object.keys(row).filter(k => !skip.has(k)).sort(utf16Compare);
}
function renderMatrix(ec7a) {
  const matrix = req(ec7a, 'matrix', 'experiment summary');
  for (const split of SPLITS) {
    const rows = req(matrix, split, 'matrix'); const t = document.getElementById('matrix-' + split);
    if (!Array.isArray(rows) || rows.length !== 4) throw new IntegrityError(6, 'V-SCHEMA', 'matrix ' + split + ' is not four rows');
    const cols = metricColumns(rows[0]);
    t.replaceChildren(el('thead', {}, el('tr', {}, ['Experiment', 'Definition', ...cols].map(h => el('th', { text: h })))),
      el('tbody', {}, rows.map(r => el('tr', {}, [el('td', { class: 'exp ' + r.experiment, text: r.experiment }), el('td', { text: r.definition }),
        ...cols.map(k => el('td', { class: 'mono', text: typeof r[k] === 'number' ? (Number.isInteger(r[k]) ? String(r[k]) : fmtNum(r[k])) : (r[k] === null ? '—' : String(r[k])) }))]))));
  }
  const acc = ec7a.accounting || {};
  if (typeof acc.internal_label_for_state_1 === 'string') document.getElementById('internal-label').textContent = acc.internal_label_for_state_1;
  const design = ec7a.design || {}; const facts = [];
  if (design.windows && typeof design.windows === 'object') for (const s of ['TRAIN', 'VALIDATION', 'TEST']) if (design.windows[s]) facts.push([s, `${design.windows[s].start} … ${design.windows[s].end}`]);
  if (ec7a.seeds && Array.isArray(ec7a.seeds.cohort)) facts.push(['Training seeds', ec7a.seeds.cohort.join(', ')]);
  if (typeof acc.bps_per_leg === 'number') facts.push(['Transaction cost', (acc.bps_per_leg / 100) + '% of traded value per purchase or sale (modelled)']);
  if (typeof acc.periods_per_year === 'number') facts.push(['Annualisation', acc.periods_per_year + ' periods per year']);
  document.getElementById('overview-facts').replaceChildren(...facts.map(([k, v]) => el('li', {}, [el('b', { text: k }), document.createTextNode(v)])));
  document.getElementById('matrix').hidden = false;
}
function renderBaselines(ec7c) {
  for (const split of SPLITS) {
    const rows = req(ec7c, split, 'baselines payload'); const t = document.getElementById('baselines-' + split);
    if (!Array.isArray(rows) || rows.length !== 7) throw new IntegrityError(6, 'V-SCHEMA', 'baselines ' + split + ' is not seven rows');
    const cols = Object.keys(rows[0]).filter(k => typeof rows[0][k] !== 'object' && k !== 'baseline_id' && k !== 'split').sort(utf16Compare);
    t.replaceChildren(el('thead', {}, el('tr', {}, ['Baseline', ...cols].map(h => el('th', { text: h })))),
      el('tbody', {}, rows.map(r => el('tr', {}, [el('td', { class: 'mono', text: req(r, 'baseline_id', 'baseline row') }),
        ...cols.map(k => el('td', { class: 'mono', text: typeof r[k] === 'number' ? (Number.isInteger(r[k]) ? String(r[k]) : fmtNum(r[k])) : (r[k] === null ? '—' : String(r[k])) }))]))));
  }
  document.getElementById('baselines').hidden = false;
}
function renderPerSeed(ec7b) {
  for (const split of SPLITS) {
    const rows = req(ec7b, split, 'per-seed payload'); const t = document.getElementById('perseed-' + split);
    if (!Array.isArray(rows) || rows.length !== 20) throw new IntegrityError(6, 'V-SCHEMA', 'per-seed ' + split + ' is not twenty rows');
    const cols = ['total_return', 'sharpe_ratio', 'max_drawdown', 'n_legs', 'total_cost_fraction', 'final_position'];
    t.replaceChildren(el('thead', {}, el('tr', {}, ['Experiment', 'Seed', ...cols].map(h => el('th', { text: h })))),
      el('tbody', {}, rows.map(r => el('tr', {}, [el('td', { class: 'exp ' + r.experiment, text: r.experiment }), el('td', { class: 'mono', text: String(r.seed) }),
        ...cols.map(k => el('td', { class: 'mono', text: r[k] === null || r[k] === undefined ? '—' : (Number.isInteger(r[k]) ? String(r[k]) : fmtNum(r[k])) }))]))));
  }
  document.getElementById('perseed').hidden = false;
}
function renderValTest(ec7d, notice) {
  const ranking = req(ec7d, 'ranking', 'validation-test payload'); const t = document.getElementById('ranking-table');
  t.replaceChildren(el('thead', {}, el('tr', {}, ['Rank', 'VALIDATION', 'TEST'].map(h => el('th', { text: h })))),
    el('tbody', {}, [0, 1, 2, 3].map(i => el('tr', {}, [el('td', { class: 'mono', text: String(i + 1) }), el('td', { class: 'exp ' + ranking.VALIDATION[i], text: ranking.VALIDATION[i] }), el('td', { class: 'exp ' + ranking.TEST[i], text: ranking.TEST[i] })]))));
  if (typeof ec7d.note === 'string') document.getElementById('valtest-note').textContent = ec7d.note;
  const box = document.getElementById('rq1-box'); const rq1 = ec7d.rq1_test;
  if (isPS(rq1) && rq1.status === 'INCLUDED') {
    const v = rq1.value; const exps = EXPERIMENTS.filter(e => e in v); const keys = exps.length ? Object.keys(v[exps[0]]).filter(k => k !== 'of').sort(utf16Compare) : [];
    const tbl = el('table', { class: 'results' }, [el('thead', {}, el('tr', {}, ['Experiment', ...keys].map(h => el('th', { text: h })))),
      el('tbody', {}, exps.map(e => el('tr', {}, [el('td', { class: 'exp ' + e, text: e }), ...keys.map(k => el('td', { class: 'mono', text: `${v[e][k]} / ${v[e].of}` }))])))]);
    box.replaceChildren(tbl, el('p', { class: 'footnote', text: 'Descriptive recomputation from the included frozen tables; not an inferential statistic.' }));
  } else box.replaceChildren(withheldNotice(notice));
  document.getElementById('valtest').hidden = false;
}
function renderProvenance(m2, projections) {
  const es = m2.evidence_set, st = m2.study_evidence_status, ds = m2.dashboard_set;
  const rows = [['Project id', ds.project_id], ['Contract', `${ds.contract.family} ${ds.contract.version}`], ['Binding identity', ds.binding_sha256], ['Dashboard-set identifier (integrity value only)', m2.dashboard_set_id],
    ['Evidence-set identifier', ds.evidence_set_id], ['Content-gate profile identifier', ds.content_gate_profile_id], ['Baseline manifest SHA-256', es.baseline_manifest_sha256], ['Dataset SHA-256', es.dataset_sha256],
    ['TEST execution commit', es.test_execution_commit], ['TEST consumed', String(es.test_consumed)], ['Study status (historical)', `${st.scientific_phase} · ${st.test_evaluation} · ${st.independent_review}`],
    ['Producer', `${m2.producer.name} ${m2.producer.version} (${short(m2.producer.source_commit, 12)})`]];
  for (const p of projections) rows.push([p.physical_copy_id, p.status === 'INCLUDED' ? `${p.status} · ${short(p.sha256, 16)} · ${p.size_bytes} bytes` : `${p.status} · ${p.reason_codes.join(', ')}`]);
  document.getElementById('prov-table').replaceChildren(...rows.map(([k, v]) => el('tr', {}, [el('td', { text: k }), el('td', { class: 'mono', text: String(v) })])));
}

/* ---------- presentation mode + nav ---------- */
function setupPresentation() {
  const params = new URLSearchParams(location.search);
  const apply = on => { document.body.classList.toggle('presentation', on); const u = new URL(location.href); if (on) u.searchParams.set('presentation', '1'); else u.searchParams.delete('presentation'); history.replaceState(null, '', u); };
  apply(params.get('presentation') === '1');
  document.getElementById('presentation-toggle').addEventListener('click', () => apply(!document.body.classList.contains('presentation')));
  const obs = new IntersectionObserver(entries => { for (const en of entries) if (en.isIntersecting) document.querySelectorAll('.site-nav a').forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + en.target.id)); }, { rootMargin: '-40% 0px -55% 0px' });
  document.querySelectorAll('section.section').forEach(s => obs.observe(s));
}

/* ---------- boot ---------- */
async function main() {
  setPanelLabels(); setupPresentation(); hideResultSections();
  let m2;
  try { m2 = await fetchEntryPoint(); }
  catch (err) { if (err instanceof NoPackageError) { renderNoPackage(err.message); return; } renderIntegrityError(err); return; }
  try {
    const projections = await verifyEntryPoint(m2);
    for (const p of projections) { PROJECTIONS[p.logical_payload_id] = p; if (p.status === 'INCLUDED') PAYLOADS[p.logical_payload_id] = await fetchVerifiedPayload(p); }
    const withheld = 'Not published: redistribution decision pending.';
    renderClaims(PAYLOADS['EC-11'], m2.notices);
    if (PAYLOADS['EC-7a']) renderMatrix(PAYLOADS['EC-7a']); else document.getElementById('matrix').replaceChildren(withheldNotice(withheld)), document.getElementById('matrix').hidden = false;
    if (PAYLOADS['EC-7c']) renderBaselines(PAYLOADS['EC-7c']); else document.getElementById('baselines').replaceChildren(withheldNotice(withheld)), document.getElementById('baselines').hidden = false;
    if (PAYLOADS['EC-7b']) renderPerSeed(PAYLOADS['EC-7b']); else document.getElementById('perseed').replaceChildren(withheldNotice(withheld)), document.getElementById('perseed').hidden = false;
    if (PAYLOADS['EC-7d']) renderValTest(PAYLOADS['EC-7d'], withheld); else document.getElementById('valtest').replaceChildren(withheldNotice(withheld)), document.getElementById('valtest').hidden = false;
    renderProvenance(m2, projections);
    document.getElementById('no-package').hidden = true;
    document.getElementById('package-badge').textContent = 'EVIDENCE PACKAGE VERIFIED (dashboard subset)';
    document.body.dataset.ready = '1';
  } catch (err) {
    renderIntegrityError(err instanceof IntegrityError ? err : new IntegrityError('?', 'V-INPUT', err.message));
    console.error(err);
  }
}
document.addEventListener('DOMContentLoaded', main);
