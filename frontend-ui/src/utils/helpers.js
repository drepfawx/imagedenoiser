export const FILTER_COLORS = {
  gaussian: '#3b82f6',
  median: '#10b981',
  bilateral: '#f59e0b',
  nlm: '#ef4444',
  cnn: '#8b5cf6',
};

export const FIXED_FILTER_ORDER = ['gaussian', 'median', 'bilateral', 'nlm', 'cnn'];

export const SHORT_NAME = (full = '') =>
  full.replace(' Filter', '').replace(' Denoiser', '')
    .replace('Non-Local Means (NLM)', 'NLM').replace('PyTorch CNN', 'CNN');

export const formatRuntime = (ms) => {
  if (ms == null) return '—';
  if (ms >= 1000) {
    const sec = ms / 1000;
    return sec < 10 ? `${sec.toFixed(2)} s` : `${sec.toFixed(1)} s`;
  }
  return ms < 1 ? `${ms.toFixed(3)} ms` : `${ms.toFixed(1)} ms`;
};

export const getBestFilter = (levels, metric) => {
  if (!levels?.length) return null;
  const fids = Object.keys(levels[0]?.results || {});
  const avgP = {};
  fids.forEach(f => {
    const vals = levels.map(l => l.results[f]?.[metric]).filter(v => v != null);
    avgP[f] = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  });

  return fids.reduce((b, f) => {
    if (avgP[f] == null) return b;
    if (avgP[b] == null) return f;
    if (metric === 'time_ms') {
      return avgP[f] < avgP[b] ? f : b; // Lower is better
    } else {
      return avgP[f] > avgP[b] ? f : b; // Higher is better
    }
  }, fids[0]);
};

export const getBestFilterByUtility = (levels) => {
  if (!levels?.length) return null;
  const fids = Object.keys(levels[0]?.results || {});
  if (!fids.length) return null;

  const avgPSNR = {};
  const avgSSIM = {};
  const avgTime = {};

  fids.forEach(fid => {
    const psnrs = levels.map(l => l.results[fid]?.psnr).filter(v => v != null);
    const ssims = levels.map(l => l.results[fid]?.ssim).filter(v => v != null);
    const times = levels.map(l => l.results[fid]?.time_ms).filter(v => v != null);

    avgPSNR[fid] = psnrs.length ? psnrs.reduce((a, b) => a + b, 0) / psnrs.length : 0;
    avgSSIM[fid] = ssims.length ? ssims.reduce((a, b) => a + b, 0) / ssims.length : 0;
    avgTime[fid] = times.length ? times.reduce((a, b) => a + b, 0) / times.length : 0;
  });

  const psnrsList = Object.values(avgPSNR);
  const ssimsList = Object.values(avgSSIM);
  const timesList = Object.values(avgTime);

  const minP = Math.min(...psnrsList);
  const maxP = Math.max(...psnrsList);
  const minS = Math.min(...ssimsList);
  const maxS = Math.max(...ssimsList);
  const minT = Math.min(...timesList);
  const maxT = Math.max(...timesList);

  let bestFid = fids[0];
  let maxUtility = -Infinity;

  fids.forEach(fid => {
    const normP = maxP > minP ? (avgPSNR[fid] - minP) / (maxP - minP) : 1.0;
    const normS = maxS > minS ? (avgSSIM[fid] - minS) / (maxS - minS) : 1.0;
    const normT = maxT > minT ? (maxT - avgTime[fid]) / (maxT - minT) : 1.0;

    const utility = 0.4 * normP + 0.4 * normS + 0.2 * normT;

    if (utility > maxUtility) {
      maxUtility = utility;
      bestFid = fid;
    }
  });

  return bestFid;
};

export const getBestAvgValue = (levels, fid, metric) => {
  if (!levels?.length || !fid) return null;
  const vals = levels.map(l => l.results[fid]?.[metric]).filter(v => v != null);
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
};

export function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = src;
  });
}

export const API_URL = '/api/process';
export const FILTER_API_URL = '/api/apply-filter';
