// app.js — orchestration: FSM, capture gate, live coaching, guardrails, overlay, results.
import * as cam from './camera.js';
import * as tracker from './tracker.js';

const API = 'https://daka-backend-9bfz.onrender.com';

const FILTER_CSS = {
  'Vivid':         'saturate(1.5) contrast(1.12)',
  'Vivid Warm':    'saturate(1.45) contrast(1.1) sepia(0.22)',
  'Vivid Cool':    'saturate(1.4) contrast(1.1) brightness(1.03) hue-rotate(6deg)',
  'Dramatic':      'contrast(1.3) saturate(0.9) brightness(0.95)',
  'Dramatic Warm': 'contrast(1.3) saturate(0.95) brightness(0.95) sepia(0.25)',
  'Dramatic Cool': 'contrast(1.32) saturate(0.85) brightness(0.95) hue-rotate(8deg)',
  'Silvertone':    'grayscale(1) sepia(0.25) contrast(1.05) brightness(1.05)',
  'Noir':          'grayscale(1) contrast(1.45) brightness(0.92)',
};

const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

/* ── State ── */
let fsm = 'HOME';
let analysis = null;                 // backend result
let capturedImg = null;              // keeper Image
let currentFilter = 'Vivid', filterActive = false;
let hasCamera = false;

let refSig = null;                   // reference frame signature (moved/blocked guardrail)
let scanPitch = 90;                  // device pitch at scan time (tilt fulfilment baseline)
let loopId = null;
let lastSigCheck = 0, sigSim = 1;

// smoothed subject
const sm = { bx: 0.5, by: 0.66, size: 0.4, has: false };
let lostSince = 0, placedSince = 0, readySince = 0, activeSince = 0;
let placed = false;
const satis = {};                    // hysteresis: id -> currently satisfied
const skipped = new Set();
let activeId = null;
let tiltDone = false;

/* ── helpers ── */
function show(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  $(id).classList.add('active');
}
let toastTimer;
function toast(msg) {
  const t = $('toast'); t.textContent = msg; t.classList.add('show');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove('show'), 2200);
}
function esc(s) { return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c])); }
function vibrate(p) { if (navigator.vibrate) navigator.vibrate(p); }

/* ── Camera state visibility ── */
function setStage(s) {
  fsm = s;
  const inCam = ['IDLE', 'ANALYZING', 'GATE', 'PLACE', 'COACH', 'READY'].includes(s);
  if (inCam) show('camera');
  $('camIdle').style.display   = s === 'IDLE' ? 'flex' : 'none';
  $('camLoading').classList.toggle('active', s === 'ANALYZING');
  $('gateBlock').classList.toggle('show', s === 'GATE');
  const coachUI = (s === 'PLACE' || s === 'COACH' || s === 'READY');
  $('aiOverlay').classList.toggle('on', coachUI);
  $('sceneBadgeTop').style.display = coachUI ? 'block' : 'none';
  $('coachBar').classList.toggle('show', coachUI);
  $('camLevel').style.display = coachUI ? 'flex' : 'none';
  $('cue').classList.toggle('show', s === 'PLACE' || s === 'COACH');
  $('poseNow').classList.toggle('show', s === 'READY');
  $('readyShutter').classList.toggle('show', s === 'READY');
  if (!coachUI) { $('cue').classList.remove('show'); }
}

/* ── Stage 1: scan the scene ── */
async function scanScene() {
  const blob = await cam.grabAnalysisBlob($('video'));
  if (!blob) { toast('Camera not ready yet'); return; }
  setStage('ANALYZING');
  try {
    const fd = new FormData(); fd.append('file', blob, 'photo.jpg');
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 60000);
    let data;
    try {
      const res = await fetch(API + '/analyze', { method: 'POST', body: fd, signal: controller.signal });
      try { data = await res.json(); } catch (e) { throw new Error('Bad response from server'); }
      if (!res.ok) throw new Error(data.error || data.message || ('Server error (' + res.status + ')'));
      if (data.error || !data.lighting) throw new Error(data.error || 'Bad response from server');
    } finally { clearTimeout(timer); }

    analysis = data;
    // Capture gate: lighting + steadiness
    if (data.lighting.quality === 'Poor') { openGate('Not enough light — reposition and try a brighter spot.'); return; }
    if (data.blurry) { openGate('Hold steady — that shot was blurry. Tap to try again.'); return; }

    // Begin coaching
    $('sceneBadgeTop').textContent = data.scene_type || 'Scene';
    refSig = cam.frameSignature($('video'));
    scanPitch = cam.getOrientation().pitch;
    resetCoaching();
    setStage('PLACE');
    startLoop();
  } catch (e) {
    setStage('IDLE');
    toast(e.name === 'AbortError' ? 'Server took too long — it may be waking up. Try again.' : (e.message || 'Analysis failed'));
  }
}

function openGate(msg) { $('gateMsg').textContent = msg; setStage('GATE'); }

function resetCoaching() {
  sm.has = false; placed = false; placedSince = 0; readySince = 0; lostSince = 0;
  activeId = null; activeSince = 0; tiltDone = false;
  skipped.clear();
  for (const k in satis) delete satis[k];
  sigSim = 1; lastSigCheck = 0;
}

/* ── Object-fit: cover transform (normalised frame coords → screen px) ── */
function coverTransform(video, canvas) {
  const W = canvas.clientWidth, H = canvas.clientHeight;
  canvas.width = W; canvas.height = H;
  const vw = video.videoWidth || W, vh = video.videoHeight || H;
  const scale = Math.max(W / vw, H / vh);
  const dw = vw * scale, dh = vh * scale;
  const ox = (W - dw) / 2, oy = (H - dh) / 2;
  return { W, H, PX: (nx) => ox + nx * dw, PY: (ny) => oy + ny * dh, S: (n) => n * dh };
}

/* ── Overlay drawing ── */
function drawOverlay(tf, target, green, markerDY) {
  const ctx = $('aiOverlay').getContext('2d');
  ctx.clearRect(0, 0, tf.W, tf.H);
  // footprint + dotted vertical box
  const fx = tf.PX(target.x), fy = tf.PY(target.y) + (markerDY || 0);
  const rw = tf.S(0.09), rh = tf.S(0.028);
  const col = green ? '#66bb6a' : '#C57A52';
  ctx.save();
  ctx.strokeStyle = col; ctx.lineWidth = 3;
  ctx.fillStyle = green ? 'rgba(102,187,106,0.22)' : 'rgba(197,122,82,0.18)';
  ctx.beginPath(); ctx.ellipse(fx, fy, rw, rh, 0, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  const boxH = tf.S(target.size || 0.6), hw = rw * 0.9;
  ctx.setLineDash([7, 7]); ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(fx - hw, fy); ctx.lineTo(fx - hw, fy - boxH);
  ctx.moveTo(fx + hw, fy); ctx.lineTo(fx + hw, fy - boxH);
  ctx.moveTo(fx - hw, fy - boxH); ctx.lineTo(fx + hw, fy - boxH);
  ctx.stroke(); ctx.setLineDash([]);
  ctx.restore();
  // subject dot
  if (sm.has) {
    const x = tf.PX(sm.bx), y = tf.PY(sm.by);
    ctx.save();
    ctx.fillStyle = green ? 'rgba(102,187,106,0.95)' : 'rgba(255,255,255,0.92)';
    ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.3)'; ctx.lineWidth = 2; ctx.stroke();
    ctx.restore();
  }
}

function setCue(label, arrow) {
  $('cueLabel').textContent = label;
  const a = $('cueArrow');
  a.textContent = arrow ? { left: '←', right: '→', up: '↑', down: '↓' }[arrow] : '';
  a.style.display = arrow ? 'block' : 'none';
}

/* ── Correction evaluation (hysteresis) ── */
function evalSat(id, value, target, inner, outer) {
  const err = Math.abs(value - target);
  const nowSat = satis[id] ? err <= outer : err <= inner;
  satis[id] = nowSat;
  return { sat: nowSat, err };
}

/* ── Main loop ── */
function startLoop() { if (!loopId) loopId = requestAnimationFrame(loop); }
function stopLoop() { if (loopId) cancelAnimationFrame(loopId); loopId = null; }

function loop(ts) {
  loopId = requestAnimationFrame(loop);
  const video = $('video');
  const tf = coverTransform(video, $('aiOverlay'));
  const target = (analysis && analysis.target) || { x: 0.667, y: 0.667, size: 0.6 };
  const o = cam.getOrientation();

  // MediaPipe unavailable → degrade to manual framing + shoot (never trap the user)
  if (tracker.hasFailed()) {
    if (fsm !== 'READY') setStage('READY');
    $('aiOverlay').getContext('2d').clearRect(0, 0, tf.W, tf.H);
    updateLevelSlider(o.roll);
    return;
  }

  // ── subject tracking ──
  let subject = tracker.isReady() ? tracker.detect(video, ts) : null;
  if (subject && subject.count >= 1 && subject.base) {
    if (!sm.has) { sm.bx = subject.base.x; sm.by = subject.base.y; sm.size = subject.size; }
    sm.bx += (subject.base.x - sm.bx) * 0.4;
    sm.by += (subject.base.y - sm.by) * 0.4;
    sm.size += (subject.size - sm.size) * 0.4;
    sm.has = true; lostSince = 0;
  } else {
    if (!lostSince) lostSince = ts;
    if (ts - lostSince > 700) sm.has = false;
  }

  // ── guardrails ──
  if (ts - lastSigCheck > 300) { lastSigCheck = ts; sigSim = cam.signatureSimilarity(refSig, cam.frameSignature(video)); }
  let guard = null;
  if (sigSim < 0.7) guard = 'Camera moved / blocked';
  else if (subject && subject.count > 1) guard = 'One person only';
  else if (!sm.has && tracker.isReady()) guard = 'Step into frame';

  // marker vertical anchor from pitch delta (responds to tilt; keeps marker on the real spot)
  const markerDY = clamp((o.pitch - scanPitch) * (tf.H * 0.006), -tf.H * 0.25, tf.H * 0.25);

  if (guard) {
    setCue(guard, null);
    $('cue').classList.add('show');
    $('poseNow').classList.remove('show');
    drawOverlay(tf, target, false, markerDY);
    updateLevelSlider(o.roll);
    return;
  }

  // ── placement check (coarse) ──
  const inZone = sm.has &&
    Math.abs(sm.bx - target.x) < 0.10 &&
    Math.abs(sm.by - target.y) < 0.12 &&
    Math.abs(sm.size - target.size) < 0.18;

  if (fsm === 'PLACE') {
    setCue(sm.has ? 'Stand on the marker' : 'Step into frame', null);
    if (inZone) {
      if (!placedSince) placedSince = ts;
      if (ts - placedSince > 400) { placed = true; vibrate(15); setStage('COACH'); }
    } else { placedSince = 0; }
    drawOverlay(tf, target, inZone, markerDY);
    updateLevelSlider(o.roll);
    return;
  }

  // ── coaching / ready ──
  // corrections
  const h = evalSat('horizontal', sm.bx, target.x, 0.05, 0.09);
  const d = evalSat('distance', sm.size, target.size, 0.12, 0.18);
  const l = evalSat('level', o.roll, 0, 3, 6);
  // tilt (only if backend flagged dead space); fulfilled by pitch moving in the asked direction
  const tiltWanted = analysis && analysis.tilt_hint && analysis.tilt_hint !== 'ok';
  if (tiltWanted && !tiltDone) {
    const delta = o.pitch - scanPitch;
    if ((analysis.tilt_hint === 'up' && delta >= 8) || (analysis.tilt_hint === 'down' && delta <= -8)) tiltDone = true;
  }

  const corr = [];
  if (!skipped.has('horizontal')) corr.push({ id: 'horizontal', sat: h.sat, err: h.err / 0.09, label: (sm.bx < target.x ? 'Move right' : 'Move left'), arrow: (sm.bx < target.x ? 'right' : 'left') });
  if (!skipped.has('distance')) corr.push({ id: 'distance', sat: d.sat, err: d.err / 0.18, label: (sm.size < target.size ? 'Come forward' : 'Step back'), arrow: null });
  if (tiltWanted && !skipped.has('tilt')) corr.push({ id: 'tilt', sat: tiltDone, err: tiltDone ? 0 : 0.7, label: (analysis.tilt_hint === 'up' ? 'Tilt up' : 'Tilt down'), arrow: (analysis.tilt_hint === 'up' ? 'up' : 'down') });
  if (!skipped.has('level')) corr.push({ id: 'level', sat: l.sat, err: l.err / 6, label: 'Level the shot', arrow: null });

  const outstanding = corr.filter(c => !c.sat).sort((a, b) => b.err - a.err);
  const allSat = outstanding.length === 0;

  updateLevelSlider(o.roll);
  drawOverlay(tf, target, inZone || allSat, markerDY);

  if (allSat) {
    if (fsm !== 'READY') { setStage('READY'); vibrate([12, 40, 12]); }
    return;
  }

  if (fsm === 'READY') setStage('COACH');   // a correction regressed
  const next = outstanding[0];
  if (next.id !== activeId) { activeId = next.id; activeSince = ts; $('skipBtn').classList.remove('show'); }
  setCue(next.label, next.arrow);
  $('cue').classList.add('show');
  // per-tip skip after a few seconds of no progress
  if (ts - activeSince > 4000) $('skipBtn').classList.add('show');
}

function skipActive() {
  if (activeId) { skipped.add(activeId); $('skipBtn').classList.remove('show'); activeSince = performance.now(); }
}

function updateLevelSlider(roll) {
  const dot = $('camLevelDot'), val = $('camLevelVal');
  if (!dot || !val) return;
  const p = clamp(((roll / 30) + 1) * 50, 0, 100);
  const level = Math.abs(roll) < 3;                 // loosened from <1° for hand-holding
  dot.style.left = p + '%';
  dot.classList.toggle('on', level);
  val.classList.toggle('on', level);
  val.textContent = Math.round(Math.abs(roll)) + '°';
}

/* ── Stage: take the keeper photo → results ── */
async function takePhoto() {
  const dataURL = await cam.grabKeeper($('video'));
  if (!dataURL) { toast('Camera not ready'); return; }
  stopLoop();
  capturedImg = new Image();
  capturedImg.src = dataURL;
  $('resultPhoto').src = dataURL;
  renderResults(analysis);
  show('results');
}

function renderResults(r) {
  currentFilter = FILTER_CSS[r.filter] ? r.filter : 'Vivid';
  filterActive = false;
  updateFilterUI();
  $('sceneBadge').textContent = r.scene_type || 'Scene';
  const pills = (r.hashtags || []).map(h => '<div class="pill" data-tag="' + esc(h) + '">' + esc(h) + '</div>').join('');
  $('cards').innerHTML = pills
    ? '<div class="card"><div class="card-title">Hashtags <span class="ch">tap to copy</span></div><div class="pills">' + pills + '</div><div class="copy-all" id="copyAll">Copy all</div></div>'
    : '';
  $('cards').querySelectorAll('.pill').forEach(p => p.addEventListener('click', () => copyText(p.dataset.tag, 'Copied!')));
  const ca = $('copyAll');
  if (ca) ca.addEventListener('click', () => copyText((r.hashtags || []).join(' '), 'All copied!'));
}

function updateFilterUI() {
  $('resultPhoto').style.filter = filterActive ? FILTER_CSS[currentFilter] : '';
  const btn = $('filterBtn');
  btn.textContent = filterActive ? currentFilter : 'Tap to preview filter';
  btn.classList.toggle('active', filterActive);
  $('saveBtn').textContent = filterActive ? 'Save with filter' : 'Save';
}

/* ── Filters (pixel-baked) + watermark ── */
function applyFilters(d, css) {
  const re = /([\w-]+)\(([^)]+)\)/g; let m;
  while ((m = re.exec(css))) {
    const v = parseFloat(m[2]);
    if (m[1] === 'brightness') fB(d, v);
    else if (m[1] === 'saturate') fS(d, v);
    else if (m[1] === 'contrast') fC(d, v);
    else if (m[1] === 'sepia') fSe(d, v);
    else if (m[1] === 'hue-rotate') fH(d, v);
    else if (m[1] === 'grayscale') fG(d, v);
  }
}
function fB(d, v) { for (let i = 0; i < d.length; i += 4) { d[i]*=v; d[i+1]*=v; d[i+2]*=v; } }
function fC(d, v) { for (let i = 0; i < d.length; i += 4) { d[i]=(d[i]-128)*v+128; d[i+1]=(d[i+1]-128)*v+128; d[i+2]=(d[i+2]-128)*v+128; } }
function fS(d, v) { for (let i = 0; i < d.length; i += 4) { const l=0.2126*d[i]+0.7152*d[i+1]+0.0722*d[i+2]; d[i]=l+(d[i]-l)*v; d[i+1]=l+(d[i+1]-l)*v; d[i+2]=l+(d[i+2]-l)*v; } }
function fSe(d, a) { const n=1-a; for (let i=0;i<d.length;i+=4){ const r=d[i],g=d[i+1],b=d[i+2]; d[i]=n*r+a*(0.393*r+0.769*g+0.189*b); d[i+1]=n*g+a*(0.349*r+0.686*g+0.168*b); d[i+2]=n*b+a*(0.272*r+0.534*g+0.131*b); } }
function fG(d, a) { const n=1-a; for (let i=0;i<d.length;i+=4){ const gr=0.2126*d[i]+0.7152*d[i+1]+0.0722*d[i+2]; d[i]=n*d[i]+a*gr; d[i+1]=n*d[i+1]+a*gr; d[i+2]=n*d[i+2]+a*gr; } }
function fH(d, deg) {
  const r=deg*Math.PI/180, c=Math.cos(r), s=Math.sin(r);
  const m0=0.213+c*0.787-s*0.213, m1=0.715-c*0.715-s*0.715, m2=0.072-c*0.072+s*0.928;
  const m3=0.213-c*0.213+s*0.143, m4=0.715+c*0.285+s*0.140, m5=0.072-c*0.072-s*0.283;
  const m6=0.213-c*0.213-s*0.787, m7=0.715-c*0.715+s*0.715, m8=0.072+c*0.928+s*0.072;
  for (let i=0;i<d.length;i+=4){ const R=d[i],G=d[i+1],B=d[i+2]; d[i]=R*m0+G*m1+B*m2; d[i+1]=R*m3+G*m4+B*m5; d[i+2]=R*m6+G*m7+B*m8; }
}
function drawWatermark(ctx, w, h) {
  const fs = Math.max(13, Math.round(w * 0.028));
  ctx.save();
  ctx.font = '600 ' + fs + 'px Georgia, "Noto Serif SC", serif';
  ctx.textAlign = 'right'; ctx.textBaseline = 'alphabetic';
  ctx.shadowColor = 'rgba(0,0,0,0.5)'; ctx.shadowBlur = fs * 0.35; ctx.shadowOffsetY = 1;
  ctx.fillStyle = 'rgba(255,255,255,0.9)';
  const pad = Math.round(w * 0.03);
  ctx.fillText('Shot with 打卡AI', w - pad, h - pad);
  ctx.restore();
}
function bakeBlob() {
  return new Promise(resolve => {
    const cv = document.createElement('canvas');
    cv.width = capturedImg.naturalWidth; cv.height = capturedImg.naturalHeight;
    const ctx = cv.getContext('2d');
    ctx.drawImage(capturedImg, 0, 0);
    if (filterActive && FILTER_CSS[currentFilter]) {
      const id = ctx.getImageData(0, 0, cv.width, cv.height);
      applyFilters(id.data, FILTER_CSS[currentFilter]);
      ctx.putImageData(id, 0, 0);
    }
    drawWatermark(ctx, cv.width, cv.height);
    cv.toBlob(b => resolve(b), 'image/jpeg', 0.95);
  });
}
async function shareOrSave(useShare) {
  if (!capturedImg) return;
  const blob = await bakeBlob();
  const tags = (analysis && analysis.hashtags || []).join(' ');
  const file = new File([blob], 'daka.jpg', { type: 'image/jpeg' });
  if (navigator.canShare && navigator.canShare({ files: [file] })) {
    try { await navigator.share(useShare ? { files: [file], text: tags } : { files: [file] }); return; }
    catch (e) { if (e.name === 'AbortError') return; }
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = 'daka.jpg';
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast('Saved');
}
async function copyText(text, msg) { try { await navigator.clipboard.writeText(text); toast(msg); } catch (e) { toast('Copy failed'); } }

/* ── Start / wiring ── */
async function begin() {
  cam.initOrientation();            // must run in the gesture (iOS permission)
  tracker.loadTracker();            // lazy MediaPipe load in the background
  hasCamera = await cam.startCamera($('video'));
  if (!hasCamera) { toast('Camera unavailable — allow camera access'); return; }
  setStage('IDLE');
}

function rescan() { stopLoop(); setStage('IDLE'); }

$('startBtn').addEventListener('click', begin);
$('shutter').addEventListener('click', scanScene);
$('gateScanAgain').addEventListener('click', rescan);
$('rescanBtn').addEventListener('click', rescan);
$('skipBtn').addEventListener('click', skipActive);
$('readyShutter').addEventListener('click', takePhoto);
$('filterBtn').addEventListener('click', () => { filterActive = !filterActive; updateFilterUI(); });
$('shareBtn').addEventListener('click', () => shareOrSave(true));
$('saveBtn').addEventListener('click', () => shareOrSave(false));
$('scanAgain').addEventListener('click', () => { if (hasCamera) { setStage('IDLE'); } else { show('home'); } });
