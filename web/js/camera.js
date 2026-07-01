// camera.js — hi-res stream, capture (analysis + keeper), device orientation, frame signature.

let stream = null;

export async function startCamera(video) {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 } },
      audio: false,
    });
    video.srcObject = stream;
    try { await video.play(); } catch (e) { /* autoplay attr covers it */ }
    return true;
  } catch (e) {
    return false;
  }
}

export function stopCamera() {
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
}

function scaledSize(w, h, cap) {
  const m = Math.max(w, h);
  if (m <= cap) return [w, h];
  const r = cap / m;
  return [Math.round(w * r), Math.round(h * r)];
}

// Small, fast frame for /analyze.
export function grabAnalysisBlob(video) {
  return new Promise(resolve => {
    if (!video.videoWidth) return resolve(null);
    const [w, h] = scaledSize(video.videoWidth, video.videoHeight, 1024);
    const c = document.createElement('canvas'); c.width = w; c.height = h;
    c.getContext('2d').drawImage(video, 0, 0, w, h);
    c.toBlob(b => resolve(b), 'image/jpeg', 0.7);
  });
}

// Hi-res keeper: full-sensor still via ImageCapture where supported (Android Chrome),
// else the full video frame. Returns a dataURL.
export async function grabKeeper(video) {
  try {
    const track = stream && stream.getVideoTracks && stream.getVideoTracks()[0];
    if (track && window.ImageCapture) {
      const ic = new ImageCapture(track);
      const blob = await ic.takePhoto();
      return await blobToDataURL(blob);
    }
  } catch (e) { /* fall through to full-frame capture */ }
  if (!video.videoWidth) return null;
  const c = document.createElement('canvas');
  c.width = video.videoWidth; c.height = video.videoHeight;
  c.getContext('2d').drawImage(video, 0, 0);
  return c.toDataURL('image/jpeg', 0.95);
}

function blobToDataURL(blob) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result);
    r.onerror = rej;
    r.readAsDataURL(blob);
  });
}

// ── Device orientation (roll = gamma, pitch = beta) ──
const orient = { roll: 0, pitch: 90, active: false };
export function getOrientation() { return orient; }

// Must be invoked from a user gesture on iOS (caller ensures).
export function initOrientation() {
  const DOE = window.DeviceOrientationEvent;
  const begin = () => { orient.active = true; window.addEventListener('deviceorientation', onOrient); };
  if (DOE && typeof DOE.requestPermission === 'function') {
    return DOE.requestPermission().then(p => { if (p === 'granted') begin(); }).catch(() => {});
  } else if (DOE) { begin(); }
  return Promise.resolve();
}
function onOrient(e) {
  if (e.gamma == null) return;
  orient.roll = e.gamma;
  orient.pitch = (e.beta == null ? 90 : e.beta);
}

// ── Frame signature ("moved/blocked" guardrail): tiny luminance histogram ──
const SIG_N = 16;
const _sigCanvas = document.createElement('canvas');
_sigCanvas.width = 48; _sigCanvas.height = 48;

export function frameSignature(video) {
  if (!video.videoWidth) return null;
  const ctx = _sigCanvas.getContext('2d');
  ctx.drawImage(video, 0, 0, 48, 48);
  const d = ctx.getImageData(0, 0, 48, 48).data;
  const hist = new Float32Array(SIG_N);
  for (let i = 0; i < d.length; i += 4) {
    const lum = 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2];
    hist[Math.min(SIG_N - 1, (lum / 256 * SIG_N) | 0)]++;
  }
  const total = 48 * 48;
  for (let i = 0; i < SIG_N; i++) hist[i] /= total;
  return hist;
}

// Bhattacharyya coefficient: 1 = identical, →0 = very different.
export function signatureSimilarity(a, b) {
  if (!a || !b) return 1;
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.sqrt(a[i] * b[i]);
  return s;
}
