/* Evidence-package loader for the read-only dashboard.
   Reads only same-origin files under ./data/ (the contract-controlled dashboard location).
   Fails closed: nothing scientific is rendered unless the entry point verifies. */
'use strict';

const CONTRACT = {
  family: 'rl-research-evidence-contract',
  supportedMajor: 1,
  // Reviewed compiled-binding identities embedded as constants at build time.
  // Empty until a compiled public binding has been independently reviewed and approved:
  // with an empty list every package fails closed at step 2 (no reviewed binding constant).
  bindingAllowlist: [],
  dataRoot: 'data/',
  entryPoint: 'provenance.json',
  dashboardDescriptors: ['EC-7a', 'EC-7b', 'EC-7c', 'EC-7d', 'EC-11'],
  projectionKeys: ['content_gates', 'logical_payload_id', 'path', 'physical_copy_id', 'reason_codes', 'sha256', 'size_bytes', 'status'],
};

class NoPackageError extends Error { constructor(msg) { super(msg); this.name = 'NoPackageError'; } }
class IntegrityError extends Error {
  constructor(step, code, detail) { super(`step ${step} ${code}: ${detail}`); this.name = 'IntegrityError'; this.step = step; this.code = code; }
}

/* ---------- canonical JSON (RFC 8785 profile used by the contract) ---------- */
function utf16Compare(a, b) {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) { const d = a.charCodeAt(i) - b.charCodeAt(i); if (d !== 0) return d; }
  return a.length - b.length;
}
function canonicalize(x) {
  if (x === null || typeof x === 'boolean') return JSON.stringify(x);
  if (typeof x === 'number') { if (!Number.isFinite(x)) throw new IntegrityError(3, 'V-SCHEMA', 'non-finite number in preimage'); return JSON.stringify(x); }
  if (typeof x === 'string') return JSON.stringify(x);
  if (Array.isArray(x)) return '[' + x.map(canonicalize).join(',') + ']';
  if (typeof x === 'object') {
    const keys = Object.keys(x).sort(utf16Compare);
    return '{' + keys.map(k => JSON.stringify(k) + ':' + canonicalize(x[k])).join(',') + '}';
  }
  throw new IntegrityError(3, 'V-SCHEMA', 'unsupported value in preimage');
}
async function sha256Hex(bytes) {
  if (!(window.isSecureContext && window.crypto && window.crypto.subtle)) {
    throw new IntegrityError(5, 'V-INPUT', 'Web Crypto is unavailable (serve over https or localhost); verification cannot run, nothing is rendered');
  }
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest)).map(b => b.toString(16).padStart(2, '0')).join('');
}

/* ---------- entry point ---------- */
async function fetchEntryPoint() {
  let r;
  try { r = await fetch(CONTRACT.dataRoot + CONTRACT.entryPoint, { cache: 'no-cache' }); }
  catch (e) { throw new NoPackageError('entry point unreachable: ' + e.message); }
  if (r.status === 404) throw new NoPackageError('no entry point at ' + CONTRACT.dataRoot + CONTRACT.entryPoint);
  if (!r.ok) throw new IntegrityError(1, 'V-ENTRY', 'entry point returned HTTP ' + r.status);
  let m2;
  try { m2 = await r.json(); } catch (e) { throw new IntegrityError(1, 'V-ENTRY', 'entry point is not JSON'); }
  return m2;
}

function requireKeys(obj, keys, step, code, what) {
  if (obj === null || typeof obj !== 'object' || Array.isArray(obj)) throw new IntegrityError(step, code, what + ' is not an object');
  for (const k of keys) if (!(k in obj)) throw new IntegrityError(step, code, what + ' lacks required key ' + k);
}

/* Steps 2–4 of the consumer sequence: contract family / major version / binding constant,
   dashboard_set_id recomputation, descriptor-projection validation. */
async function verifyEntryPoint(m2) {
  requireKeys(m2, ['dashboard_set', 'dashboard_set_id', 'evidence_set', 'producer', 'study_evidence_status', 'public_provenance', 'notices'], 1, 'V-ENTRY', 'entry point');
  const ds = m2.dashboard_set;
  requireKeys(ds, ['binding_sha256', 'content_gate_profile_id', 'contract', 'descriptors', 'evidence_set_id', 'project_id'], 2, 'V-CONTRACT', 'dashboard_set');
  if (ds.contract.family !== CONTRACT.family) throw new IntegrityError(2, 'V-CONTRACT', 'unsupported contract family');
  const major = parseInt(String(ds.contract.version).split('.')[0], 10);
  if (!(major === CONTRACT.supportedMajor)) throw new IntegrityError(2, 'V-VERSION', 'unsupported contract major version ' + ds.contract.version);
  if (!CONTRACT.bindingAllowlist.includes(ds.binding_sha256)) throw new IntegrityError(2, 'V-CONTRACT', 'binding identity is not a reviewed constant of this build');

  const recomputed = await sha256Hex(new TextEncoder().encode(canonicalize(ds)));
  if (recomputed !== m2.dashboard_set_id) throw new IntegrityError(3, 'V-DSTHASH', 'dashboard_set_id does not recompute');

  const projections = ds.descriptors;
  if (!Array.isArray(projections)) throw new IntegrityError(4, 'V-DESCRIPTOR', 'descriptors is not an array');
  const expectedOrder = [['EC-7a', 'EC-7a#1'], ['EC-7b', 'EC-7b#1'], ['EC-7c', 'EC-7c#1'], ['EC-7d', 'EC-7d#1'], ['EC-11', 'EC-11#dashboard']];
  if (projections.length !== expectedOrder.length) throw new IntegrityError(4, 'V-DESCRIPTOR', 'unexpected number of dashboard descriptors');
  projections.forEach((p, i) => {
    const keys = Object.keys(p).sort(utf16Compare);
    if (keys.join(',') !== CONTRACT.projectionKeys.join(',')) throw new IntegrityError(4, 'V-DESCRIPTOR', 'projection keys of ' + (p.physical_copy_id || '?') + ' are not the eight required keys');
    if (p.logical_payload_id !== expectedOrder[i][0] || p.physical_copy_id !== expectedOrder[i][1]) throw new IntegrityError(4, 'V-DESCRIPTOR', 'descriptor order or identity mismatch at index ' + i);
    if (!['INCLUDED', 'WITHHELD', 'BLOCKED'].includes(p.status)) throw new IntegrityError(4, 'V-DESCRIPTOR', 'invalid status for ' + p.physical_copy_id);
    if (p.status === 'INCLUDED') {
      if (typeof p.path !== 'string' || !p.path.startsWith('dashboard/data/') || p.path.includes('..')) throw new IntegrityError(4, 'V-ENTRY', 'included path outside dashboard/data/ for ' + p.physical_copy_id);
      if (typeof p.sha256 !== 'string' || p.sha256.length !== 64 || !Number.isInteger(p.size_bytes)) throw new IntegrityError(4, 'V-DESCRIPTOR', 'included descriptor without hash or size: ' + p.physical_copy_id);
      if (!Array.isArray(p.reason_codes) || p.reason_codes.length !== 0) throw new IntegrityError(4, 'V-DESCRIPTOR', 'included descriptor carries reason codes: ' + p.physical_copy_id);
    } else {
      if (p.path !== null || p.sha256 !== null || p.size_bytes !== null) throw new IntegrityError(4, 'V-DESCRIPTOR', 'withheld descriptor with physical fields: ' + p.physical_copy_id);
      if (!Array.isArray(p.reason_codes) || p.reason_codes.length === 0) throw new IntegrityError(4, 'V-DESCRIPTOR', 'withheld descriptor without reason codes: ' + p.physical_copy_id);
    }
  });
  const claims = projections[4];
  if (claims.status !== 'INCLUDED') throw new IntegrityError(4, 'V-DESCRIPTOR', 'the claims payload must be INCLUDED');
  return projections;
}

/* Step 5: fetch exactly the INCLUDED projections' paths; verify size and SHA-256 before parsing. */
async function fetchVerifiedPayload(projection) {
  const rel = projection.path.slice('dashboard/'.length); // dashboard/data/x.json → data/x.json relative to this page
  const r = await fetch(rel, { cache: 'no-cache' });
  if (!r.ok) throw new IntegrityError(5, 'V-MISSING', projection.physical_copy_id + ' could not be fetched (HTTP ' + r.status + ')');
  const bytes = new Uint8Array(await r.arrayBuffer());
  if (bytes.length !== projection.size_bytes) throw new IntegrityError(5, 'V-DSTHASH', projection.physical_copy_id + ' size differs from its descriptor');
  const digest = await sha256Hex(bytes);
  if (digest !== projection.sha256) throw new IntegrityError(5, 'V-DSTHASH', projection.physical_copy_id + ' SHA-256 differs from its descriptor');
  try { return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes)); }
  catch (e) { throw new IntegrityError(6, 'V-SCHEMA', projection.physical_copy_id + ' is not valid JSON'); }
}

/* ---------- formatting ---------- */
function fmtSigned(x, d = 4) { if (x === null || x === undefined || Number.isNaN(x)) return '—'; const s = x.toFixed(d); return (x > 0 ? '+' : '') + s; }
function fmtNum(x, d = 4) { if (x === null || x === undefined || Number.isNaN(x)) return '—'; return x.toFixed(d); }
function fmtPM(mean, std, d = 4, signed = true) { return (signed ? fmtSigned(mean, d) : fmtNum(mean, d)) + ' ± ' + fmtNum(std, d); }
function fmtInt(x) { return x === null || x === undefined ? '—' : Math.round(x).toLocaleString('en-US'); }
function short(hash, n = 12) { return hash ? hash.slice(0, n) + '…' : '—'; }

function el(tag, attrs = {}, children = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of [].concat(children)) if (c !== null && c !== undefined) e.append(c);
  return e;
}
