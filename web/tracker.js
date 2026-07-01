// tracker.js — MediaPipe PoseLandmarker wrapper, lazy-loaded from CDN.

let landmarker = null, loading = false, failed = false;
const V = '0.10.14';

export function isReady() { return !!landmarker; }
export function hasFailed() { return failed; }

export async function loadTracker() {
  if (landmarker || loading) return;
  loading = true;
  try {
    const vision = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@' + V);
    const fileset = await vision.FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@' + V + '/wasm'
    );
    const model = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
    const opts = (delegate) => ({
      baseOptions: { modelAssetPath: model, delegate },
      runningMode: 'VIDEO',
      numPoses: 2,
    });
    try { landmarker = await vision.PoseLandmarker.createFromOptions(fileset, opts('GPU')); }
    catch (e) { landmarker = await vision.PoseLandmarker.createFromOptions(fileset, opts('CPU')); }
  } catch (e) {
    failed = true;
  } finally {
    loading = false;
  }
}

function avg(a, b) { if (!a || !b) return null; return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }; }

// Per-frame result: { count, base:{x,y}|null, size, center:{x,y}|null, headroom } or null.
// All coords are normalised 0..1 in the video's intrinsic frame, top-left origin.
export function detect(video, tsMs) {
  if (!landmarker || !video.videoWidth) return null;
  let res;
  try { res = landmarker.detectForVideo(video, tsMs); } catch (e) { return null; }
  const poses = (res && res.landmarks) || [];
  if (!poses.length) return { count: 0, base: null, size: 0, center: null, headroom: 1 };

  const lm = poses[0];
  const base = avg(lm[27], lm[28]) || avg(lm[23], lm[24]) || lm[0];   // ankles → hips → nose
  const head = lm[0] || lm[11] || base;                               // nose → shoulder → base
  const center = avg(lm[11], lm[12]) || avg(lm[23], lm[24]) || base;
  const size = Math.max(0, Math.min(1, base.y - head.y));             // vertical extent (distance proxy)
  return {
    count: poses.length,
    base: { x: base.x, y: base.y },
    size,
    center: { x: center.x, y: center.y },
    headroom: head.y,                                                 // 0 = head at top edge
  };
}
