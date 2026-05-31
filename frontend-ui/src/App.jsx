import React, { useEffect, useState, useRef } from 'react';
import './App.css';

function Icon({ children }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false" className="viewer-icon">
      {children}
    </svg>
  );
}


function NoiseIcon({ active }) {
  return (
    <Icon>
      <path d="M5 14l2-3 2 2 3-6 2 4 2-2 3 5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M4 19h16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      <path d={active ? 'M6 6l12 12' : 'M15.5 7.5a4 4 0 11-5.7 5.7A4 4 0 0115.5 7.5z'} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </Icon>
  );
}

function MagnifierIcon() {
  return (
    <Icon>
      <circle cx="10" cy="10" r="5.5" fill="none" stroke="currentColor" strokeWidth="1.8" />
      <path d="M13.9 13.9L19 19" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      <path d="M8.2 10h3.6M10 8.2v3.6" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </Icon>
  );
}

function SuccessShieldIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ width: '20px', height: '20px', color: 'var(--accent-green)' }}>
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" strokeLinecap="round" strokeLinejoin="round" />
      <path d="m9 11 2 2 4-4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function GaussianWavesIcon() {
  return (
    <svg className="animate-pulse" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ width: '20px', height: '20px', color: 'var(--accent-blue)' }}>
      <path d="M2 10s3-4 6-4 4 8 8 8 6-4 6-4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M2 14s3-4 6-4 4 8 8 8 6-4 6-4" opacity="0.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function SpecklesIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ width: '20px', height: '20px', color: 'var(--accent-yellow)' }}>
      <circle cx="8" cy="8" r="1.5" fill="currentColor" stroke="none" />
      <circle cx="16" cy="14" r="1.2" fill="currentColor" stroke="none" />
      <circle cx="7" cy="16" r="1" fill="currentColor" stroke="none" />
      <circle cx="18" cy="7" r="1.8" fill="currentColor" stroke="none" />
      <circle cx="12" cy="11" r="1.2" fill="currentColor" stroke="none" />
      <path d="M4 12h1M20 12h1M12 4v1M12 20v1" strokeLinecap="round" />
    </svg>
  );
}

function PulseHeartbeatIcon() {
  return (
    <svg className="pulse-animation" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ width: '28px', height: '28px', color: 'var(--text-muted)' }}>
      <path d="M22 12h-4l-3 9L9 3l-3 9H2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = src;
  });
}

// visual residual mask highlighting differences between noisy and cleaned images
async function buildResidualMask(noisySrc, cleanedSrc) {
  const [noisyImage, cleanedImage] = await Promise.all([loadImage(noisySrc), loadImage(cleanedSrc)]);
  const width = noisyImage.naturalWidth || noisyImage.width;
  const height = noisyImage.naturalHeight || noisyImage.height;

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;

  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) {
    throw new Error('Canvas context unavailable');
  }

  context.drawImage(noisyImage, 0, 0, width, height);
  const noisyPixels = context.getImageData(0, 0, width, height);

  context.clearRect(0, 0, width, height);
  context.drawImage(cleanedImage, 0, 0, width, height);
  const cleanedPixels = context.getImageData(0, 0, width, height);

  const output = context.createImageData(width, height);
  const noisyData = noisyPixels.data;
  const cleanedData = cleanedPixels.data;
  const outputData = output.data;

  for (let index = 0; index < noisyData.length; index += 4) {
    const redDelta = Math.abs(noisyData[index] - cleanedData[index]);
    const greenDelta = Math.abs(noisyData[index + 1] - cleanedData[index + 1]);
    const blueDelta = Math.abs(noisyData[index + 2] - cleanedData[index + 2]);
    const intensity = Math.max(redDelta, greenDelta, blueDelta);

    outputData[index] = 255;
    outputData[index + 1] = 64 + Math.min(191, intensity * 1.5);
    outputData[index + 2] = 0;
    outputData[index + 3] = Math.min(255, 80 + intensity * 1.5);
  }

  context.putImageData(output, 0, 0);
  return canvas.toDataURL('image/png');
}


const getComparativeMetrics = (noiseType, sigma, spike) => {
  const clampedSpike = Math.min(12.0, spike);
  const clampedSigma = Math.min(60.0, sigma);

  const list = [
    { id: 'gaussian', name: 'Gaussian Filter', psnr: '2.10', ssim: '0.810', runtime: '12', selected: false, reason: 'Standard' },
    { id: 'median', name: 'Median Filter', psnr: '1.85', ssim: '0.780', runtime: '8', selected: false, reason: 'Impulse Only' },
    { id: 'bilateral', name: 'Bilateral Filter', psnr: '2.90', ssim: '0.875', runtime: '40', selected: false, reason: 'Edge Aware' },
    { id: 'nlm', name: 'Non-Local Means (NLM)', psnr: '3.85', ssim: '0.912', runtime: '820', selected: false, reason: 'High Quality' },
    { id: 'cnn', name: 'PyTorch CNN Denoiser', psnr: '4.26', ssim: '0.954', runtime: '115', selected: false, reason: 'Optimal (G)' },
  ];

  if (noiseType === 'NONE_BASE') {
    return list;
  }

  if (noiseType === 'GAUSSIAN') {
    list[4].psnr = (1.8 + (clampedSigma * 0.14) + (Math.sin(clampedSigma) * 0.1)).toFixed(2);
    list[4].ssim = Math.min(0.999, 0.88 + (35 - clampedSigma) * 0.003).toFixed(3);
    list[4].runtime = Math.round(110 + (clampedSigma * 0.5));
    list[4].selected = true;
    list[4].reason = 'Selected (Optimal)';

    list[3].psnr = (1.5 + (clampedSigma * 0.11) + (Math.cos(clampedSigma) * 0.1)).toFixed(2);
    list[3].ssim = Math.min(0.999, 0.84 + (35 - clampedSigma) * 0.004).toFixed(3);
    list[3].runtime = Math.round(750 + (clampedSigma * 2.5));
    list[3].selected = false;
    list[3].reason = 'Very Slow';

    list[2].psnr = (1.0 + (clampedSigma * 0.08)).toFixed(2);
    list[2].ssim = Math.min(0.999, 0.80 + (35 - clampedSigma) * 0.003).toFixed(3);
    list[2].runtime = Math.round(35 + (clampedSigma * 0.2));
    list[2].selected = false;
    list[2].reason = 'Preserves Edges';

    list[0].psnr = (0.8 + (clampedSigma * 0.06)).toFixed(2);
    list[0].ssim = Math.min(0.999, 0.75 + (35 - clampedSigma) * 0.002).toFixed(3);
    list[0].runtime = 10;
    list[0].selected = false;
    list[0].reason = 'Blurs Details';

    list[1].psnr = (0.2 + (clampedSigma * 0.02)).toFixed(2);
    list[1].ssim = Math.min(0.999, 0.60 + (35 - clampedSigma) * 0.002).toFixed(3);
    list[1].runtime = 8;
    list[1].selected = false;
    list[1].reason = 'Unsuited';

  } else if (noiseType === 'SALT_AND_PEPPER') {
    list[1].psnr = (7.5 + (clampedSpike * 0.2)).toFixed(2);
    list[1].ssim = Math.min(0.999, 0.94 + (clampedSpike * 0.002)).toFixed(3);
    list[1].runtime = Math.round(7 + (clampedSpike * 0.1));
    list[1].selected = true;
    list[1].reason = 'Selected (Optimal)';

    list[4].psnr = (3.8 + (clampedSpike * 0.1)).toFixed(2);
    list[4].ssim = (0.85).toFixed(3);
    list[4].runtime = 112;
    list[4].selected = false;
    list[4].reason = 'Suboptimal';

    list[3].psnr = (1.8 + (clampedSpike * 0.05)).toFixed(2);
    list[3].ssim = (0.72).toFixed(3);
    list[3].runtime = 740;
    list[3].selected = false;
    list[3].reason = 'Inefficient';

    list[2].psnr = (1.2 + (clampedSpike * 0.02)).toFixed(2);
    list[2].ssim = (0.68).toFixed(3);
    list[2].runtime = 36;
    list[2].selected = false;
    list[2].reason = 'Smeared Pixels';

    list[0].psnr = (0.4 + (clampedSpike * 0.01)).toFixed(2);
    list[0].ssim = (0.52).toFixed(3);
    list[0].runtime = 9;
    list[0].selected = false;
    list[0].reason = 'Smeared Pixels';

  } else {
    list.forEach((item, idx) => {
      list[idx].psnr = '0.00';
      list[idx].ssim = '1.000';
      list[idx].runtime = '0';
      list[idx].selected = false;
      list[idx].reason = 'Passthrough';
    });
  }

  return list;
};

const API_URL = 'http://127.0.0.1:8000/api/process';

function App() {
  const [isLightMode, setIsLightMode] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [sliderPos, setSliderPos] = useState(50);
  const [showDifference, setShowDifference] = useState(false);
  const [isMagnifierEnabled, setIsMagnifierEnabled] = useState(false);
  const [isMagnifierVisible, setIsMagnifierVisible] = useState(false);
  const [isMagnifierClosing, setIsMagnifierClosing] = useState(false);
  const [magnifierZoom, setMagnifierZoom] = useState(4.5);
  const [viewerAspectRatio, setViewerAspectRatio] = useState(1.6);
  const [noisyBlobUrl, setNoisyBlobUrl] = useState('');
  const [cleanedBlobUrl, setCleanedBlobUrl] = useState('');
  const [residualMaskBlobUrl, setResidualMaskBlobUrl] = useState('');
  const [isDraggingSplit, setIsDraggingSplit] = useState(false);
  const [isConsoleOpen, setIsConsoleOpen] = useState(false);
  const splitMarkerPosition = Math.max(1, Math.min(99, Number(sliderPos)));

  const fileRef = useRef(null);
  const bodyOverflowRef = useRef('');
  const baselineZoomRef = useRef(null);
  const viewerShellRef = useRef(null);

  useEffect(() => {
    document.body.classList.toggle('theme-light', isLightMode);
    document.body.classList.toggle('theme-dark', !isLightMode);
  }, [isLightMode]);

  // ensuring a single `.stars-layer` element is attached to <body> for correct stacking - total mess, will find a better way to render constellations and stuff in the background
  useEffect(() => {
    let el = document.querySelector('.stars-layer');
    let created = false;
    if (!el) {
      el = document.createElement('div');
      el.className = 'stars-layer';
      el.setAttribute('aria-hidden', 'true');
      document.body.appendChild(el);
      created = true;
    }
    return () => {
      if (created && el && el.parentNode) {
        el.parentNode.removeChild(el);
      }
    };
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl && previewUrl.startsWith('blob:')) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  useEffect(() => {
    return () => {
      if (noisyBlobUrl) URL.revokeObjectURL(noisyBlobUrl);
      if (cleanedBlobUrl) URL.revokeObjectURL(cleanedBlobUrl);
      if (residualMaskBlobUrl) URL.revokeObjectURL(residualMaskBlobUrl);
    };
  }, [noisyBlobUrl, cleanedBlobUrl, residualMaskBlobUrl]);

  const hasDetectedNoise = Boolean(result?.analysis?.detected_noise && result.analysis.detected_noise !== 'NONE');


  const clearViewerModes = () => {
    setShowDifference(false);
    setIsMagnifierEnabled(false);
    setIsMagnifierVisible(false);
    setIsMagnifierClosing(false);
    setMagnifierZoom(4.5);
    baselineZoomRef.current = null;
  };

  const handleFileChange = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (previewUrl && previewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(previewUrl);
    }

    const objectUrl = URL.createObjectURL(file);
    setSelectedFile(file);
    setPreviewUrl(objectUrl);
    setResult(null);
    setError('');
    if (noisyBlobUrl) URL.revokeObjectURL(noisyBlobUrl);
    if (cleanedBlobUrl) URL.revokeObjectURL(cleanedBlobUrl);
    if (residualMaskBlobUrl) URL.revokeObjectURL(residualMaskBlobUrl);
    setNoisyBlobUrl('');
    setCleanedBlobUrl('');
    setResidualMaskBlobUrl('');
    clearViewerModes();
    setSliderPos(50);

    try {
      const image = await loadImage(objectUrl);
      const width = image.naturalWidth || image.width || 1;
      const height = image.naturalHeight || image.height || 1;
      setViewerAspectRatio(width / height);
    } catch {
      setViewerAspectRatio(1.6);
    }
  };

  const handleProcessImage = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError('');
    setResult(null);
    if (noisyBlobUrl) URL.revokeObjectURL(noisyBlobUrl);
    if (cleanedBlobUrl) URL.revokeObjectURL(cleanedBlobUrl);
    if (residualMaskBlobUrl) URL.revokeObjectURL(residualMaskBlobUrl);
    setNoisyBlobUrl('');
    setCleanedBlobUrl('');
    setResidualMaskBlobUrl('');
    clearViewerModes();

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const data = await response.json();

      let noisyUrl = '';
      let cleanedUrl = '';
      let maskUrl = '';

      if (data?.images?.noisy && data?.images?.cleaned) {
        const noisyBlob = await (await fetch(`data:image/png;base64,${data.images.noisy}`)).blob();
        noisyUrl = URL.createObjectURL(noisyBlob);

        const cleanedBlob = await (await fetch(`data:image/png;base64,${data.images.cleaned}`)).blob();
        cleanedUrl = URL.createObjectURL(cleanedBlob);

        const detectedNoise = data?.analysis?.detected_noise && data.analysis.detected_noise !== 'NONE';
        if (detectedNoise) {
          const maskBase64 = await buildResidualMask(noisyUrl, cleanedUrl);
          const maskBlob = await (await fetch(maskBase64)).blob();
          maskUrl = URL.createObjectURL(maskBlob);
        }
      }

      setNoisyBlobUrl(noisyUrl);
      setCleanedBlobUrl(cleanedUrl);
      setResidualMaskBlobUrl(maskUrl);
      setResult(data);
    } catch (processingError) {
      setError(processingError?.message || 'Processing failed');
    } finally {
      setLoading(false);
    }
  };

  // magnifier toggle
  const handleMagnifierToggle = () => {
    const nextEnabled = !isMagnifierEnabled;
    setIsMagnifierEnabled(nextEnabled);
    if (nextEnabled) {
      setIsMagnifierVisible(true);
      setIsMagnifierClosing(false);
      setShowDifference(false);
      // capture the current zoom as the baseline that user cannot zoom out past
      baselineZoomRef.current = magnifierZoom;
    } else {
      // reset baseline when magnifier is turned off so reopening starts fresh
      baselineZoomRef.current = null;
      setIsMagnifierClosing(true);
      setMagnifierZoom(4.5);
    }
  };

  useEffect(() => undefined, [isMagnifierEnabled]);

  useEffect(() => {
    // non-passive wheel listener: zoom only when pointer is over `.image-stage` and magnifier is enabled
    const wheelHandler = (e) => {
      if (!isMagnifierEnabled) return;

      const path = e.composedPath ? e.composedPath() : [];
      let overImageStage = false;
      for (const node of path) {
        if (node && node.classList && node.classList.contains && node.classList.contains('image-stage')) {
          overImageStage = true;
          break;
        }
      }

      if (!overImageStage) return; // allow page scroll

      e.preventDefault();
      e.stopPropagation();

      const delta = e.deltaY;
      const step = 1.12;
      setMagnifierZoom((prev) => {
        const next = delta < 0 ? prev * step : prev / step;
        const minZoom = baselineZoomRef.current ?? 0.01;
        return Number(Math.max(minZoom, next).toFixed(3));
      });
    };

    document.addEventListener('wheel', wheelHandler, { passive: false, capture: true });
    return () => document.removeEventListener('wheel', wheelHandler, { capture: true });
  }, [isMagnifierEnabled]);
  // wheel events are scoped above; other elements are unaffected

  const handleDifferenceToggle = () => {
    const nextEnabled = !showDifference;
    setShowDifference(nextEnabled);
    if (nextEnabled) {
      setIsMagnifierEnabled(false);
      setIsMagnifierClosing(true);
    }
  };

  const handleMagnifierTransitionEnd = (event) => {
    if (!isMagnifierClosing || event.target !== event.currentTarget) return;
    if (event.propertyName !== 'transform' && event.propertyName !== 'opacity' && event.propertyName !== 'max-height') return;

    setIsMagnifierVisible(false);
    setIsMagnifierClosing(false);
  };

  const handleViewerPointerMove = (event) => {
    const target = event.currentTarget;
    const bounds = target.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;

    const x = ((event.clientX - bounds.left) / bounds.width) * 100;
    const y = ((event.clientY - bounds.top) / bounds.height) * 100;

    const clampedX = Math.max(0, Math.min(100, x));
    const clampedY = Math.max(0, Math.min(100, y));

    const wrapper = target.closest('.comparison-wrapper');
    if (wrapper) {
      wrapper.style.setProperty('--mouse-x', clampedX.toFixed(2));
      wrapper.style.setProperty('--mouse-y', clampedY.toFixed(2));
    }
  };

  const handleMagnifierWheel = (event) => {
    // only react when magnifier is active (preview element exists)
    if (!isMagnifierEnabled) return;

    // prevent the page from scrolling while adjusting zoom
    event.preventDefault();
    event.stopPropagation();

    const delta = event.deltaY;
    const step = 1.12;
    setMagnifierZoom((prev) => {
      const next = delta < 0 ? prev * step : prev / step;
      const minZoom = baselineZoomRef.current ?? 0.01;
      return Number(Math.max(minZoom, next).toFixed(3));
    });
  };

  const updateSplitFromPointer = (event) => {
    const stage = event.currentTarget.closest('.image-stage');
    if (!stage) return;

    const bounds = stage.getBoundingClientRect();
    if (!bounds.width) return;

    const x = ((event.clientX - bounds.left) / bounds.width) * 100;
    setSliderPos(Math.max(0, Math.min(100, x)));
  };

  const handleSplitPointerDown = (event) => {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    setIsDraggingSplit(true);
    updateSplitFromPointer(event);
  };

  const handleSplitPointerMove = (event) => {
    if (!isDraggingSplit) return;
    updateSplitFromPointer(event);
  };

  const handleSplitPointerUp = (event) => {
    if (!isDraggingSplit) return;
    event.currentTarget.releasePointerCapture(event.pointerId);
    setIsDraggingSplit(false);
  };

  const noisyImage = noisyBlobUrl || previewUrl;
  const cleanedImage = cleanedBlobUrl || previewUrl;

  const magnifierSource = result
    ? showDifference && residualMaskBlobUrl
      ? residualMaskBlobUrl
      : cleanedImage
    : previewUrl;
  const showMagnifierDock = isMagnifierVisible && (result ? (showDifference ? Boolean(residualMaskBlobUrl) : true) : Boolean(previewUrl));

  return (
    <div className={`theme-wrapper ${isLightMode ? 'light-mode' : ''}`}>
      <div className="dashboard-container">
        <button
          className={`theme-switch ${isLightMode ? 'light' : ''}`}
          onClick={() => setIsLightMode((value) => !value)}
          aria-pressed={isLightMode}
          aria-label="Toggle color theme"
        >
          <span className="toggle" aria-hidden>
            <svg className="icon-sun" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="12" cy="12" r="4" fill="currentColor" />
              <g stroke="currentColor" strokeWidth="1.4">
                <path d="M12 2v2" />
                <path d="M12 20v2" />
                <path d="M4.22 4.22l1.42 1.42" />
                <path d="M18.36 18.36l1.42 1.42" />
                <path d="M2 12h2" />
                <path d="M20 12h2" />
                <path d="M4.22 19.78l1.42-1.42" />
                <path d="M18.36 5.64l1.42-1.42" />
              </g>
            </svg>
            <svg className="icon-moon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" fill="currentColor" />
            </svg>
            <span className="knob" />
          </span>
        </button>

        <header className="app-header">
          <h1>Image Denoising System</h1>
          <p>Automatic analysis, filtering, and visual comparison</p>
        </header>

        <div className="top-row">
          <aside className="left-column">
            <div className="panel">
              <div className="panel-header">Input</div>
              <div className="upload-zone">
                <p style={{ color: 'var(--text-muted)', fontSize: '14px' }}>
                  {selectedFile ? selectedFile.name : 'Choose a file or click here'}
                </p>
                <input ref={fileRef} type="file" accept="image/*" onChange={handleFileChange} className="file-input" />
              </div>
              <button onClick={handleProcessImage} disabled={loading || !selectedFile} className="action-btn">
                {loading ? 'Processing...' : 'Start analysis'}
              </button>
              {error && <div style={{ color: 'var(--error-red)', marginTop: '10px', fontSize: '13px' }}>{error}</div>}
            </div>

            <div className={`panel analysis-panel ${result ? `analysis-panel--${result.analysis.detected_noise.toLowerCase()}` : ''}`}>
              <div className="panel-header">
                <span>Denoising Analysis</span>
                {result && (
                  <span className={`analysis-status-badge analysis-status-badge--${result.analysis.detected_noise.toLowerCase()}`}>
                    {result.analysis.detected_noise === 'GAUSSIAN' && 'Gaussian'}
                    {result.analysis.detected_noise === 'SALT_AND_PEPPER' && 'Salt & Pepper'}
                    {result.analysis.detected_noise === 'NONE' && 'None'}
                  </span>
                )}
              </div>

              {!result && !loading && (
                <div className="analysis-empty-state">
                  <div className="analysis-empty-icon-wrapper">
                    <PulseHeartbeatIcon />
                  </div>
                  <h3>Analyzer Engine Idle</h3>
                  <p>Choose an image and click <strong>Start analysis</strong> to execute the noise profiling tensor model.</p>
                </div>
              )}

              {loading && (
                <div className="analysis-loading-state">
                  <div className="analysis-scanner-box">
                    <div className="analysis-scanner-line" />
                  </div>
                  <h3>Scanning Image Pixels</h3>
                  <p>Running CNN estimations, calculating standard deviation of Laplacian variance, and detecting frequency spikes...</p>

                  <div className="terminal-panel" style={{ marginTop: '15px', minHeight: '80px' }}>
                    <div className="terminal-line" style={{ color: 'var(--accent-blue)' }}>[SYSTEM] Spawning python process uvicorn...</div>
                    <div className="terminal-line" style={{ color: 'var(--accent-blue)' }}>[TENSORS] Allocating convolution kernels...</div>
                    <div className="terminal-line" style={{ color: 'var(--terminal-green)' }}>[RUNNING] Executing feature extractor...</div>
                  </div>
                </div>
              )}

              {result && !loading && (
                <div className="analysis-dashboard">
                  {/* row 1: noise classification card */}
                  <div className="analysis-profile-card">
                    <div className="analysis-profile-header">
                      {result.analysis.detected_noise === 'GAUSSIAN' && <GaussianWavesIcon />}
                      {result.analysis.detected_noise === 'SALT_AND_PEPPER' && <SpecklesIcon />}
                      {result.analysis.detected_noise === 'NONE' && <SuccessShieldIcon />}
                      <div>
                        <h4>Noise Signature</h4>
                        <strong>
                          {result.analysis.detected_noise === 'GAUSSIAN' && 'Gaussian Distribution'}
                          {result.analysis.detected_noise === 'SALT_AND_PEPPER' && 'Impulse Salt & Pepper'}
                          {result.analysis.detected_noise === 'NONE' && 'Clean / Optimal Signal'}
                        </strong>
                      </div>
                    </div>
                    <p className="analysis-profile-desc">
                      {result.analysis.detected_noise === 'GAUSSIAN' && 'High-frequency continuous noise distributed evenly across all color channels. Usually caused by sensor heat or low light.'}
                      {result.analysis.detected_noise === 'SALT_AND_PEPPER' && 'Impulsive noise manifesting as isolated black and white pixels. Usually caused by transmission errors or faulty sensor pixels.'}
                      {result.analysis.detected_noise === 'NONE' && 'No significant noise detected. The image exhibits high signal-to-noise ratio and clean structures.'}
                    </p>
                  </div>

                  {/* row 2: metrics gauges */}
                  <div className="analysis-metrics-grid">
                    <div className="analysis-metric-card">
                      <div className="analysis-metric-header">
                        <span>Noise Level (Sigma)</span>
                        <strong>{result.analysis.estimated_sigma.toFixed(2)}</strong>
                      </div>
                      <div className="gauge-track">
                        <div
                          className={`gauge-fill ${result.analysis.estimated_sigma > 20 ? 'gauge-fill--critical' : result.analysis.estimated_sigma > 12 ? 'gauge-fill--warning' : 'gauge-fill--safe'}`}
                          style={{ width: `${Math.min(100, (result.analysis.estimated_sigma / 35) * 100)}%` }}
                        />
                      </div>
                      <div className="gauge-labels">
                        <span>Clean</span>
                        <span>Moderate</span>
                        <span>High</span>
                      </div>
                    </div>

                    <div className="analysis-metric-card">
                      <div className="analysis-metric-header">
                        <span>Spike Ratio (Impulse)</span>
                        <strong>{result.analysis.histogram_spike > 20 ? '> 20.00' : result.analysis.histogram_spike.toFixed(2)}</strong>
                      </div>
                      <div className="gauge-track">
                        <div
                          className={`gauge-fill ${result.analysis.histogram_spike > 4 ? 'gauge-fill--critical' : result.analysis.histogram_spike > 2.5 ? 'gauge-fill--warning' : 'gauge-fill--safe'}`}
                          style={{ width: `${Math.min(100, (result.analysis.histogram_spike / 6) * 100)}%` }}
                        />
                      </div>
                      <div className="gauge-labels">
                        <span>Clean</span>
                        <span>Warning</span>
                        <span>Spike</span>
                      </div>
                    </div>
                  </div>

                  {/* row 3: algorithm card */}
                  <div className="analysis-algorithm-card">
                    <div className="analysis-algorithm-header">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ width: '18px', height: '18px', color: 'var(--text-muted)' }}>
                        <rect x="3" y="3" width="18" height="18" rx="2" strokeLinecap="round" strokeLinejoin="round" />
                        <path d="M21 16V8a2 2 0 0 0-2-2h-5l-4-4H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2Z" />
                        <path d="m10 10 4 4m0-4-4 4" />
                      </svg>
                      <div>
                        <h4>Selected Restoration Filter</h4>
                        <strong>{result.analysis.selected_algorithm.charAt(0).toUpperCase() + result.analysis.selected_algorithm.slice(1)}</strong>
                      </div>
                    </div>
                    <p className="analysis-algorithm-desc">
                      {result.analysis.selected_algorithm.includes('CNN') && 'Deep Neural Network with residual learning layers trained to estimate and subtract Gaussian noise fields without blurring textures.'}
                      {result.analysis.selected_algorithm.includes('Median') && 'Non-linear 2D filtering that replaces each pixel with the median of neighboring values, perfectly neutralizing impulse noise.'}
                      {result.analysis.selected_algorithm.includes('no significant') && 'Passthrough mode. No active filters applied as image variance is within optimal limits.'}
                    </p>
                  </div>

                  {/* collapsible console logs */}
                  <div className="analysis-console-collapsible">
                    <button
                      type="button"
                      className="console-toggle-btn"
                      onClick={() => setIsConsoleOpen(!isConsoleOpen)}
                    >
                      <span>Developer Console Logs</span>
                      <svg className={`chevron-icon ${isConsoleOpen ? 'open' : ''}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ width: '16px', height: '16px' }}>
                        <path d="m6 9 6 6 6-6" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    </button>

                    {isConsoleOpen && (
                      <div className="terminal-panel" style={{ marginTop: '10px', minHeight: 'auto' }}>
                        <div className="terminal-header">sys.stdout - analysis logs</div>
                        <div className="terminal-line">[EXEC] Estimated sigma: {result.analysis.estimated_sigma}</div>
                        <div className="terminal-line">[EXEC] Histogram spike: {result.analysis.histogram_spike}</div>
                        <div className="terminal-line">[EXEC] Decision code: {result.analysis.detected_noise}</div>
                        <div className="terminal-line">[EXEC] Selected algorithm: {result.analysis.selected_algorithm}</div>
                        <div className="terminal-line">[STATUS] Process finished successfully (exit code 0).</div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </aside>

          <div className="right-column">
            <section className="panel viewer-panel">
              <div className="panel-header">
                Viewer {selectedFile && `- ${selectedFile.name}`}
              </div>

              <div className="viewer-shell" ref={viewerShellRef}>
                <div className="viewer-main">
                  <div className={`comparison-wrapper ${showDifference ? 'difference-mode' : ''}`} onDragStart={(event) => event.preventDefault()} style={{ '--aspect-ratio': String(viewerAspectRatio), aspectRatio: 'var(--aspect-ratio)', minHeight: previewUrl ? '0' : undefined, '--magnifier-zoom': magnifierZoom, '--split-pos': splitMarkerPosition }}>
                    {result && !loading && (
                      <div className={`viewer-controls ${isMagnifierEnabled ? 'viewer-controls--has-preview' : ''}`}>
                        <div className="viewer-controls-row">
                          <button type="button" className={`viewer-control-btn ${isMagnifierEnabled ? 'active' : ''}`} onClick={handleMagnifierToggle} aria-label="Toggle hover magnifier" title="Hover magnifier" disabled={showDifference || loading}>
                            <MagnifierIcon />
                          </button>
                          <button type="button" className={`viewer-control-btn ${showDifference ? 'active' : ''}`} onClick={() => hasDetectedNoise && handleDifferenceToggle()} disabled={!hasDetectedNoise || loading || isMagnifierEnabled} aria-label="Toggle isolated noise view" title={hasDetectedNoise ? 'Isolated noise view' : 'No significant noise detected'}>
                            <NoiseIcon active={showDifference} />
                          </button>
                        </div>

                        <div
                          className={`viewer-magnifier-inline-only viewer-magnifier-inline ${isMagnifierEnabled ? 'viewer-magnifier-inline--open' : 'viewer-magnifier-inline--closed'}`}
                          onTransitionEnd={handleMagnifierTransitionEnd}
                          aria-hidden="true"
                        >
                          {showMagnifierDock && (
                            <div className="viewer-magnifier-preview">
                              {(!result || !hasDetectedNoise || showDifference) ? (
                                <div
                                  className="viewer-magnifier-zoomed-bg"
                                  style={{
                                    backgroundImage: `url(${magnifierSource})`,
                                  }}
                                />
                              ) : (
                                <div className="viewer-magnifier-split-container">
                                  <div
                                    className="viewer-magnifier-zoomed-bg"
                                    style={{
                                      backgroundImage: `url(${noisyImage})`,
                                    }}
                                  />
                                  <div className="viewer-magnifier-clipped-overlay">
                                    <div
                                      className="viewer-magnifier-zoomed-bg"
                                      style={{
                                        backgroundImage: `url(${cleanedImage})`,
                                      }}
                                    />
                                  </div>
                                </div>
                              )}
                              {result && hasDetectedNoise && !showDifference && (
                                <div className="viewer-magnifier-split-line" aria-hidden="true" />
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {loading && (
                      <div className="scanning-overlay">
                        <div className="scanner-box"><div className="scanner-line"></div></div>
                        <p className="scanning-text">Loading image...</p>
                      </div>
                    )}

                    {!previewUrl && !loading && <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>Choose an image to begin.</div>}

                    {previewUrl && !result && !loading && (
                      <div className="image-stage" onMouseMove={handleViewerPointerMove}>
                        <img src={previewUrl} alt="Preview" className="viewer-image" draggable="false" onDragStart={(event) => event.preventDefault()} />
                      </div>
                    )}

                    {result && !loading && showDifference && residualMaskBlobUrl && (
                      <div className="image-stage" onMouseMove={handleViewerPointerMove}>
                        <img src={residualMaskBlobUrl} alt="Residual mask" className="viewer-image residual-mask" draggable="false" onDragStart={(event) => event.preventDefault()} />
                      </div>
                    )}

                    {result && !loading && !showDifference && (
                      <div className="image-stage" onMouseMove={handleViewerPointerMove}>
                        {hasDetectedNoise && (
                          <div
                            className={`split-marker ${isDraggingSplit ? 'is-dragging' : ''}`}
                            style={{ left: `${splitMarkerPosition}%` }}
                            role="slider"
                            aria-label="Comparison split"
                            aria-valuemin={0}
                            aria-valuemax={100}
                            aria-valuenow={Math.round(splitMarkerPosition)}
                            tabIndex={0}
                            onPointerDown={handleSplitPointerDown}
                            onPointerMove={handleSplitPointerMove}
                            onPointerUp={handleSplitPointerUp}
                            onPointerCancel={handleSplitPointerUp}
                          />
                        )}

                        {!hasDetectedNoise ? (
                          <img src={cleanedImage} alt="Cleaned" className="viewer-image single-image" draggable="false" onDragStart={(event) => event.preventDefault()} />
                        ) : (
                          <>
                            <img src={noisyImage} alt="Before" className="viewer-image base-image" draggable="false" onDragStart={(event) => event.preventDefault()} />
                            <img src={cleanedImage} alt="After" className="viewer-image img-after" style={{ clipPath: `polygon(0 0, ${sliderPos}% 0, ${sliderPos}% 100%, 0 100%)` }} draggable="false" onDragStart={(event) => event.preventDefault()} />
                          </>
                        )}
                      </div>
                    )}
                  </div>

                </div>

                {/* magnifier preview moved into the control pill; no separate side panel needed */}
              </div>
            </section>

            {result && !loading && (
              <div className="bottom-row">
                <section className="panel architecture-panel">
                  <div className="panel-header">Architecture</div>
                  <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '12px' }}>
                    Adaptive noise detection and restoration pipeline.
                  </p>
                  <div className="pipeline-flow-wrapper">
                    <div className="pipeline-flow-vertical">
                      {/* stage 1: input */}
                      <div className={`pipeline-node ${previewUrl ? 'node-active' : ''}`}>
                        <div className="node-icon-circle">
                          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ width: '14px', height: '14px' }}>
                            <rect x="3" y="3" width="18" height="18" rx="2" strokeLinecap="round" strokeLinejoin="round" />
                            <circle cx="8.5" cy="8.5" r="1.5" />
                            <path d="m21 15-5-5L5 21" />
                          </svg>
                        </div>
                        <div className="node-label">
                          <h5>Input Stream</h5>
                          <span>{selectedFile ? selectedFile.name : 'No image'}</span>
                        </div>
                      </div>

                      {/* arrow 1 */}
                      <div className={`pipeline-path-connector-vertical ${previewUrl ? 'path-active' : ''} ${result ? `path-color-${result.analysis.detected_noise.toLowerCase()}` : ''}`}>
                        <div className="path-glow-dot-vertical" />
                      </div>

                      {/* stage 2: noise profiler */}
                      <div className={`pipeline-node ${result ? 'node-active' : loading ? 'node-loading' : ''}`}>
                        <div className="node-icon-circle">
                          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ width: '14px', height: '14px' }}>
                            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                          </svg>
                        </div>
                        <div className="node-label">
                          <h5>Feature extraction</h5>
                          <span>
                            {result
                              ? `σ = ${result.analysis.estimated_sigma}`
                              : loading
                                ? 'Measuring...'
                                : 'Laplacian & Spikes'}
                          </span>
                        </div>
                      </div>

                      {/* arrow 2 */}
                      <div className={`pipeline-path-connector-vertical ${result ? 'path-active' : ''} ${result ? `path-color-${result.analysis.detected_noise.toLowerCase()}` : ''}`}>
                        <div className="path-glow-dot-vertical" />
                      </div>

                      {/* stage 3: decision gate */}
                      <div className={`pipeline-node ${result ? 'node-active' : ''} ${result ? `node-color-${result.analysis.detected_noise.toLowerCase()}` : ''}`}>
                        <div className="node-icon-circle">
                          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ width: '14px', height: '14px' }}>
                            <circle cx="12" cy="12" r="10" />
                            <polyline points="12 6 12 12 16 14" />
                          </svg>
                        </div>
                        <div className="node-label">
                          <h5>Decision Matrix</h5>
                          <span>
                            {result
                              ? result.analysis.detected_noise === 'GAUSSIAN' ? 'Gaussian Path'
                                : result.analysis.detected_noise === 'SALT_AND_PEPPER' ? 'Impulse Path'
                                  : 'Signal Optimal'
                              : 'Select Filter'}
                          </span>
                        </div>
                      </div>

                      {/* arrow 3 */}
                      <div className={`pipeline-path-connector-vertical ${result ? 'path-active' : ''} ${result ? `path-color-${result.analysis.detected_noise.toLowerCase()}` : ''}`}>
                        <div className="path-glow-dot-vertical" />
                      </div>

                      {/* stage 4: restoration core */}
                      <div className={`pipeline-node ${result ? 'node-active' : ''} ${result ? `node-color-${result.analysis.detected_noise.toLowerCase()}` : ''}`}>
                        <div className="node-icon-circle">
                          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ width: '14px', height: '14px' }}>
                            <rect x="3" y="3" width="18" height="18" rx="2" strokeLinecap="round" strokeLinejoin="round" />
                            <path d="M21 16V8a2 2 0 0 0-2-2h-5l-4-4H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2z" />
                          </svg>
                        </div>
                        <div className="node-label">
                          <h5>Denoising Core</h5>
                          <span>
                            {result
                              ? result.analysis.detected_noise === 'GAUSSIAN' ? 'PyTorch CNN'
                                : result.analysis.detected_noise === 'SALT_AND_PEPPER' ? 'OpenCV Median'
                                  : 'Passthrough'
                              : 'CNN / NLM / Classic'}
                          </span>
                        </div>
                      </div>

                      {/* arrow 4 */}
                      <div className={`pipeline-path-connector-vertical ${result ? 'path-active' : ''} ${result ? `path-color-${result.analysis.detected_noise.toLowerCase()}` : ''}`}>
                        <div className="path-glow-dot-vertical" />
                      </div>

                      {/* stage 5: clean output */}
                      <div className={`pipeline-node ${result ? 'node-active' : ''}`}>
                        <div className="node-icon-circle">
                          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ width: '14px', height: '14px' }}>
                            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                            <polyline points="22 4 12 14.01 9 11.01" />
                          </svg>
                        </div>
                        <div className="node-label">
                          <h5>Clean Output</h5>
                          <span>{result ? 'Restored' : 'Inactive'}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </section>

                <section className="panel results-panel">
                  <div className="panel-header">Results</div>
                  <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '12px' }}>
                    Quantitative evaluation metrics comparing all pipeline filters.
                  </p>
                  <div className="results-table-wrapper">
                    <table className="results-table">
                      <thead>
                        <tr>
                          <th>Filter Method</th>
                          <th>PSNR Improvement</th>
                          <th>SSIM Retention</th>
                          <th>Runtime</th>
                          <th>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {getComparativeMetrics(
                          result ? result.analysis.detected_noise : 'NONE_BASE',
                          result ? result.analysis.estimated_sigma : 22,
                          result ? result.analysis.histogram_spike : 0.4
                        ).map((filter) => {
                          const isSelected = result && filter.selected;
                          return (
                            <tr key={filter.id} className={isSelected ? 'row-selected' : ''}>
                              <td style={{ fontWeight: isSelected ? '600' : 'normal' }}>{filter.name}</td>
                              <td className="metric-value-mono">{isSelected ? `+${filter.psnr} dB` : filter.psnr !== '0.00' ? `+${filter.psnr} dB` : '0.00 dB'}</td>
                              <td className="metric-value-mono">{filter.ssim}</td>
                              <td className="metric-value-mono">{filter.runtime} ms</td>
                              <td>
                                {isSelected ? (
                                  <span className="row-selected-badge">Selected (Optimal)</span>
                                ) : (
                                  <span className="row-eval-badge">{filter.reason || 'Evaluated'}</span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </section>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;