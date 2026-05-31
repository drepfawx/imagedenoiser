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

function MonitorDisplayIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false" className="viewer-empty-icon">
      <rect x="4.5" y="5.5" width="15" height="10" rx="1.8" fill="none" stroke="currentColor" strokeWidth="1.5" />
      <path d="M9 19h6M12 15.5v3.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M8 10.2h1.8M14.2 10.2H16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M10.1 12.6h3.8" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
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

const API_URL = 'http://127.0.0.1:8000/api/process';
const FILTER_API_URL = 'http://127.0.0.1:8000/api/apply-filter';

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
  const [previewFadeActive, setPreviewFadeActive] = useState(false);
  const [frozenViewerHeight, setFrozenViewerHeight] = useState(null);
  const [lastAnalyzedFileKey, setLastAnalyzedFileKey] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [histogramData, setHistogramData] = useState(null);
  const [activeFilterId, setActiveFilterId] = useState(null);
  const activeFilterIdRef = useRef(activeFilterId);
  useEffect(() => {
    activeFilterIdRef.current = activeFilterId;
  }, [activeFilterId]);

  const [filterLoading, setFilterLoading] = useState(false);
  const [backgroundLoadingIds, setBackgroundLoadingIds] = useState([]);

  const [backgroundQueue, setBackgroundQueueState] = useState([]);
  const backgroundQueueRef = useRef([]);
  const setBackgroundQueue = (queue) => {
    backgroundQueueRef.current = queue;
    setBackgroundQueueState(queue);
  };
  const backgroundFetchSessionRef = useRef(null);
  const activeFetchesRef = useRef({});
  const filterAbortControllerRef = useRef(null);
  const resultRef = useRef(result);

  useEffect(() => {
    resultRef.current = result;
  }, [result]);

  const splitMarkerPosition = Math.max(1, Math.min(99, Number(sliderPos)));

  const selectedFileKey = selectedFile
    ? `${selectedFile.name}:${selectedFile.size}:${selectedFile.lastModified}`
    : '';
  const isCurrentFileAlreadyAnalyzed = Boolean(selectedFileKey) && selectedFileKey === lastAnalyzedFileKey;

  const fileRef = useRef(null);
  const bodyOverflowRef = useRef('');
  const baselineZoomRef = useRef(null);
  const viewerShellRef = useRef(null);
  const comparisonWrapperRef = useRef(null);

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
    return () => { if (noisyBlobUrl) URL.revokeObjectURL(noisyBlobUrl); };
  }, [noisyBlobUrl]);

  useEffect(() => {
    return () => { if (cleanedBlobUrl) URL.revokeObjectURL(cleanedBlobUrl); };
  }, [cleanedBlobUrl]);

  useEffect(() => {
    return () => { if (residualMaskBlobUrl) URL.revokeObjectURL(residualMaskBlobUrl); };
  }, [residualMaskBlobUrl]);

  const hasDetectedNoise = Boolean(result?.analysis?.detected_noise && result.analysis.detected_noise !== 'NONE');
  const clearViewerModes = () => {
    setShowDifference(false);
    setIsMagnifierEnabled(false);
    setIsMagnifierVisible(false);
    setIsMagnifierClosing(false);
    setMagnifierZoom(4.5);
    baselineZoomRef.current = null;
  };

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;

    if (files && files.length > 0) {
      const file = files[0];

      if (file.type.startsWith('image/')) {
        handleFileChange({ target: { files: [file] } });
        setError('');
      } else {
        setError('Unsupported file format. Please drop an image file (e.g., JPG, PNG).');
      }
    }
  };

  const handleFileChange = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    backgroundFetchSessionRef.current = null; // abort any running background fetches
    filterAbortControllerRef.current?.abort(); // abort active request
    setBackgroundLoadingIds([]);
    setBackgroundQueue([]);

    if (previewUrl && previewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(previewUrl);
    }

    const objectUrl = URL.createObjectURL(file);
    setSelectedFile(file);
    setLastAnalyzedFileKey('');
    setPreviewUrl(objectUrl);
    setPreviewFadeActive(true);
    setResult(null);
    setError('');
    if (noisyBlobUrl) URL.revokeObjectURL(noisyBlobUrl);
    if (cleanedBlobUrl) URL.revokeObjectURL(cleanedBlobUrl);
    if (residualMaskBlobUrl) URL.revokeObjectURL(residualMaskBlobUrl);
    setNoisyBlobUrl('');
    setCleanedBlobUrl('');
    setResidualMaskBlobUrl('');
    setHistogramData(null);
    setActiveFilterId(null);
    setFrozenViewerHeight(null);
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

  const applyCachedFilter = async (filterId, base64, maskBase64, cleanedHist) => {
    try {
      const cleanedBlob = await (await fetch(`data:image/png;base64,${base64}`)).blob();
      const newCleanedUrl = URL.createObjectURL(cleanedBlob);

      // revoke old cleaned and mask URLs
      if (cleanedBlobUrl) URL.revokeObjectURL(cleanedBlobUrl);
      if (residualMaskBlobUrl) URL.revokeObjectURL(residualMaskBlobUrl);

      let newMaskUrl = '';
      if (maskBase64) {
        try {
          const maskBlob = await (await fetch(`data:image/png;base64,${maskBase64}`)).blob();
          newMaskUrl = URL.createObjectURL(maskBlob);
        } catch (maskError) {
          console.error("Mask apply failed:", maskError);
        }
      }

      setCleanedBlobUrl(newCleanedUrl);
      setResidualMaskBlobUrl(newMaskUrl);
      setActiveFilterId(filterId);
      setFilterLoading(false);

      // Set precomputed histogram data directly
      if (resultRef.current?.images?.noisy_hist && cleanedHist) {
        setHistogramData({
          noisy: resultRef.current.images.noisy_hist,
          cleaned: cleanedHist,
        });
      }
    } catch (e) {
      console.error("Apply cached filter failed:", e);
    }
  };

  const runBackgroundQueue = async (sessionToken, uncomputedFilterIds, file) => {
    setBackgroundQueue(uncomputedFilterIds);

    while (backgroundQueueRef.current.length > 0) {
      if (backgroundFetchSessionRef.current !== sessionToken) return;

      const filterId = backgroundQueueRef.current[0];

      // check if it was already cached/fetched
      if (resultRef.current?.images?.all_cleaned?.[filterId]) {
        setBackgroundQueue(backgroundQueueRef.current.filter((id) => id !== filterId));
        continue;
      }

      const controller = new AbortController();
      filterAbortControllerRef.current = controller;

      const fetchPromise = (async () => {
        try {
          const formData = new FormData();
          formData.append('file', file);

          const response = await fetch(`${FILTER_API_URL}?filter=${encodeURIComponent(filterId)}`, {
            method: 'POST',
            body: formData,
            signal: controller.signal,
          });

          if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
          }

          const data = await response.json();
          if (backgroundFetchSessionRef.current !== sessionToken) return null;

          setResult((prev) => {
            if (!prev) return prev;
            return {
              ...prev,
              images: {
                ...prev.images,
                all_cleaned: {
                  ...prev.images.all_cleaned,
                  [filterId]: data.images.cleaned,
                },
                all_histograms: {
                  ...prev.images.all_histograms,
                  [filterId]: data.images.cleaned_hist,
                },
                all_masks: {
                  ...prev.images.all_masks,
                  [filterId]: data.images.residual_mask,
                },
              },
              filter_metrics: prev.filter_metrics.map((metric) => {
                if (metric.id === filterId) {
                  return {
                    ...metric,
                    ...data.metrics,
                  };
                }
                return metric;
              }),
            };
          });

          if (activeFilterIdRef.current === filterId) {
            await applyCachedFilter(filterId, data.images.cleaned, data.images.residual_mask, data.images.cleaned_hist);
            setFilterLoading(false);
          }

          return data.images.cleaned;
        } catch (err) {
          if (err.name === 'AbortError') {
            console.log(`Background fetch aborted for ${filterId}`);
            // No-op: handleFilterSwitch prepended the aborted filter to the queue right after the clicked filter
          } else {
            console.error("Background fetch failed for", filterId, err);
          }
          return null;
        } finally {
          delete activeFetchesRef.current[filterId];
          setBackgroundLoadingIds((prev) => prev.filter((id) => id !== filterId));
          if (filterAbortControllerRef.current === controller) {
            filterAbortControllerRef.current = null;
          }
        }
      })();

      activeFetchesRef.current[filterId] = fetchPromise;
      setBackgroundLoadingIds((prev) => [...prev, filterId]);
      setBackgroundQueue(backgroundQueueRef.current.filter((id) => id !== filterId));

      try {
        await fetchPromise;
      } catch (err) {
        // Abort error handled inside fetchPromise
      }
    }
  };

  const handleProcessImage = async () => {
    if (!selectedFile || isCurrentFileAlreadyAnalyzed) return;

    backgroundFetchSessionRef.current = null; // abort any running background fetches for the previous image
    filterAbortControllerRef.current?.abort(); // abort active request
    setBackgroundLoadingIds([]);
    setBackgroundQueue([]);

    const wrapperHeight = comparisonWrapperRef.current?.getBoundingClientRect()?.height;
    if (wrapperHeight && Number.isFinite(wrapperHeight)) {
      setFrozenViewerHeight(Math.round(wrapperHeight));
    }

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
        if (detectedNoise && data?.images?.residual_mask) {
          const maskBlob = await (await fetch(`data:image/png;base64,${data.images.residual_mask}`)).blob();
          maskUrl = URL.createObjectURL(maskBlob);
        }
      }

      setNoisyBlobUrl(noisyUrl);
      setCleanedBlobUrl(cleanedUrl);
      setResidualMaskBlobUrl(maskUrl);
      setResult(data);
      setLastAnalyzedFileKey(selectedFileKey);

      // determine which filter was auto-selected
      setActiveFilterId(data?.analysis?.best_filter_id || null);

      // set pixel intensity histograms directly from precomputed data
      if (data?.images?.noisy_hist && data?.images?.cleaned_hist) {
        setHistogramData({
          noisy: data.images.noisy_hist,
          cleaned: data.images.cleaned_hist,
        });
      }

      // start background pre-fetching for other uncomputed filters if noise is detected
      const isCleanSignal = data?.analysis?.detected_noise === 'NONE';
      if (!isCleanSignal) {
        const sessionToken = Math.random().toString(36).substring(7);
        backgroundFetchSessionRef.current = sessionToken;

        const uncomputedFilters = (data?.filter_metrics || [])
          .filter((f) => f.id !== data?.analysis?.best_filter_id && f.id !== null)
          .map((f) => f.id);

        if (uncomputedFilters.length > 0) {
          runBackgroundQueue(sessionToken, uncomputedFilters, selectedFile);
        }
      }
    } catch (processingError) {
      setError(processingError?.message || 'Processing failed');
    } finally {
      setLoading(false);
    }
  };

  // switch to a different denoising filter on the already-analyzed image
  const handleFilterSwitch = async (filterId) => {
    if (!selectedFile || !result || loading) return;

    if (filterId === activeFilterId) {
      clearViewerModes();
      setActiveFilterId(null);
      return;
    }

    // 1. check if the filtered image is cached on the client
    const cachedBase64 = result?.images?.all_cleaned?.[filterId];
    if (cachedBase64) {
      const cachedMask = result?.images?.all_masks?.[filterId];
      const cachedHist = result?.images?.all_histograms?.[filterId];
      clearViewerModes();
      await applyCachedFilter(filterId, cachedBase64, cachedMask, cachedHist);
      return;
    }

    // 2. check if it's currently fetching in the background
    const existingFetchPromise = activeFetchesRef.current[filterId];
    if (existingFetchPromise) {
      setFilterLoading(true);
      clearViewerModes();
      setActiveFilterId(filterId);
      return;
    }

    // 3. not cached and not currently fetching -> make it priority and start it
    setFilterLoading(true);
    clearViewerModes();
    setActiveFilterId(filterId);

    // find currently active fetching filter, if any
    const activeLoadingId = backgroundLoadingIds[0] || Object.keys(activeFetchesRef.current)[0];

    // put it at the front of the background queue, followed immediately by the interrupted one (if any)
    let nextQueue = backgroundQueueRef.current.filter((id) => id !== filterId);
    if (activeLoadingId && activeLoadingId !== filterId) {
      nextQueue = nextQueue.filter((id) => id !== activeLoadingId);
      nextQueue = [filterId, activeLoadingId, ...nextQueue];
    } else {
      nextQueue = [filterId, ...nextQueue];
    }

    // generate a new session token to either restart the running queue with new priority or start a new queue session if the loop wasn't running.
    const sessionToken = Math.random().toString(36).substring(7);
    backgroundFetchSessionRef.current = sessionToken;

    // abort whatever is currently fetching so this priority one starts immediately
    if (filterAbortControllerRef.current) {
      filterAbortControllerRef.current.abort();
    }

    runBackgroundQueue(sessionToken, nextQueue, selectedFile);
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
      : activeFilterId
        ? cleanedImage
        : noisyImage
    : previewUrl;
  const showMagnifierDock = isMagnifierVisible && (result ? (showDifference ? Boolean(residualMaskBlobUrl) : true) : Boolean(previewUrl));
  const isDifferenceUsable = Boolean(result && activeFilterId && residualMaskBlobUrl);

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
            <div className="panel input-panel">
              <div className="panel-header">Input</div>
              <div
                className={`upload-zone ${isDragging ? 'is-dragging' : ''}`}
                onDragEnter={handleDragEnter}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <p className="upload-file-name">
                  {selectedFile ? selectedFile.name : 'Choose a file or drag and drop here'}
                </p>
                <input ref={fileRef} type="file" accept="image/*" onChange={handleFileChange} className="file-input" />
              </div>

              <div className={`action-container ${selectedFile || error ? 'is-active' : ''}`}>
                <div className="action-wrapper">
                  {selectedFile && (
                    <button
                      onClick={handleProcessImage}
                      disabled={loading || isCurrentFileAlreadyAnalyzed}
                      className="action-btn"
                    >
                      {loading ? 'Processing...' : isCurrentFileAlreadyAnalyzed ? 'Already analyzed' : 'Start Analysis'}
                    </button>
                  )}

                  {error && (
                    <div className="upload-error-msg">
                      {error}
                    </div>
                  )}
                </div>
              </div>
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
                  <p>Choose an image and click <strong>Start Analysis</strong> to execute the noise profiling tensor model.</p>
                </div>
              )}

              {loading && (
                <div className="analysis-loading-state">
                  <div className="analysis-scanner-box">
                    <div className="analysis-scanner-line" />
                  </div>
                  <h3>Scanning Image Pixels</h3>
                  <p>Running CNN estimations, calculating standard deviation of Laplacian variance, and detecting frequency spikes...</p>
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
                        <strong>
                          {result.filter_metrics?.find(m => m.id === activeFilterId)?.name || 'Passthrough'}
                        </strong>
                      </div>
                    </div>
                    <p className="analysis-algorithm-desc">
                      {activeFilterId === 'cnn' && 'Deep Neural Network with residual learning layers trained to estimate and subtract Gaussian noise fields without blurring textures.'}
                      {activeFilterId === 'median' && 'Non-linear 2D filtering that replaces each pixel with the median of neighboring values, perfectly neutralizing impulse noise.'}
                      {activeFilterId === 'gaussian' && 'Linear 2D filter that uses a Gaussian kernel to smooth high-frequency details, effective for mild Gaussian noise.'}
                      {activeFilterId === 'bilateral' && 'Edge-preserving smoothing filter that weights pixels by spatial proximity and radiometric similarity, reducing noise while keeping sharp edges.'}
                      {activeFilterId === 'nlm' && 'Non-Local Means filtering that averages pixels based on the similarity of their neighborhoods, highly effective but computationally expensive.'}
                      {!activeFilterId && 'Passthrough mode. No active filters applied as image variance is within optimal limits.'}
                    </p>
                  </div>

                  {/* row 4: pixel intensity histogram */}
                  {histogramData && (
                    <div className="histogram-card">
                      <div className="histogram-header">
                        <span>Pixel Intensity</span>
                        <div className="histogram-legend">
                          <span className="legend-dot legend-dot--noisy" /><span>Noisy</span>
                          <span className="legend-dot legend-dot--cleaned" /><span>Cleaned</span>
                        </div>
                      </div>
                      <svg
                        className="histogram-svg"
                        viewBox={`0 0 ${histogramData.noisy.length * 5} 60`}
                        preserveAspectRatio="none"
                        aria-label="Pixel intensity histogram"
                      >
                        {/* noisy layer */}
                        {histogramData.noisy.map((val, i) => (
                          <rect
                            key={`n${i}`}
                            x={i * 5}
                            y={60 - val * 58}
                            width={4}
                            height={val * 58}
                            rx={1}
                            fill="rgba(239,68,68,0.38)"
                          />
                        ))}
                        {/* cleaned layer */}
                        {histogramData.cleaned.map((val, i) => (
                          <rect
                            key={`c${i}`}
                            x={i * 5}
                            y={60 - val * 58}
                            width={4}
                            height={val * 58}
                            rx={1}
                            fill="rgba(52,211,153,0.55)"
                          />
                        ))}
                      </svg>
                      <div className="histogram-axis">
                        <span>Shadows</span>
                        <span>Midtones</span>
                        <span>Highlights</span>
                      </div>
                    </div>
                  )}

                </div>
              )}
            </div>
          </aside>

          <div className="right-column">
            <section className="panel viewer-panel">
              <div className="panel-header">
                <span className="truncate-text">
                  Viewer {selectedFile && `- ${selectedFile.name}`}
                </span>
              </div>

              <div className="viewer-shell" ref={viewerShellRef}>
                <div className="viewer-main">
                  <div className={`comparison-wrapper ${showDifference ? 'difference-mode' : ''}`} ref={comparisonWrapperRef} onDragStart={(event) => event.preventDefault()} style={{ '--aspect-ratio': String(viewerAspectRatio), aspectRatio: 'var(--aspect-ratio)', minHeight: frozenViewerHeight ? `${frozenViewerHeight}px` : (previewUrl ? '0' : undefined), height: frozenViewerHeight ? `${frozenViewerHeight}px` : undefined, maxHeight: frozenViewerHeight ? `${frozenViewerHeight}px` : undefined, '--magnifier-zoom': magnifierZoom, '--split-pos': splitMarkerPosition }}>
                    {result && !loading && !filterLoading && (
                      <div className={`viewer-controls ${isMagnifierEnabled ? 'viewer-controls--has-preview' : ''}`}>
                        <div className="viewer-controls-row">
                          <button type="button" className={`viewer-control-btn ${isMagnifierEnabled ? 'active' : ''}`} onClick={handleMagnifierToggle} aria-label="Toggle hover magnifier" title="Hover magnifier" disabled={showDifference || loading || filterLoading}>
                            <MagnifierIcon />
                          </button>
                          <button type="button" className={`viewer-control-btn ${showDifference ? 'active' : ''}`} onClick={() => isDifferenceUsable && handleDifferenceToggle()} disabled={!isDifferenceUsable || loading || filterLoading || isMagnifierEnabled} aria-label="Toggle isolated noise view" title={isDifferenceUsable ? 'Isolated noise view' : (!activeFilterId ? 'Apply a filter to isolate differences' : 'No difference data available')}>
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
                              {(!result || !activeFilterId || showDifference) ? (
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
                              {result && activeFilterId && !showDifference && (
                                <div className="viewer-magnifier-split-line" aria-hidden="true" />
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {(loading || filterLoading) && (
                      <div className="scanning-overlay">
                        <div className="scanner-box"><div className="scanner-line"></div></div>
                      </div>
                    )}

                    {!previewUrl && !loading && (
                      <div className="viewer-empty-state">
                        <MonitorDisplayIcon />
                        <div className="viewer-empty-copy">
                          <p className="viewer-empty-title">Viewer ready.</p>
                          <p className="viewer-empty-text">Use the upload bar to load an image.</p>
                        </div>
                      </div>
                    )}

                    {previewUrl && !result && (
                      <div className="image-stage" onMouseMove={handleViewerPointerMove}>
                        <img
                          key={previewUrl}
                          src={previewUrl}
                          alt="Preview"
                          className={`viewer-image${previewFadeActive ? ' viewer-image--fade-in' : ''}${(loading || filterLoading) ? ' viewer-image--scanning' : ''}`}
                          draggable="false"
                          onDragStart={(event) => event.preventDefault()}
                          onAnimationEnd={() => setPreviewFadeActive(false)}
                        />
                      </div>
                    )}

                    {result && filterLoading && (
                      <div className="image-stage" onMouseMove={handleViewerPointerMove}>
                        <img
                          key={noisyImage}
                          src={noisyImage}
                          alt="Preview"
                          className="viewer-image viewer-image--scanning single-image"
                          draggable="false"
                          onDragStart={(event) => event.preventDefault()}
                        />
                      </div>
                    )}

                    {result && !loading && !filterLoading && showDifference && residualMaskBlobUrl && (
                      <div className="image-stage" onMouseMove={handleViewerPointerMove}>
                        <img key={cleanedImage} src={cleanedImage} alt="Cleaned Base" className="viewer-image base-image" draggable="false" onDragStart={(event) => event.preventDefault()} />
                        <img key={residualMaskBlobUrl} src={residualMaskBlobUrl} alt="Residual mask" className="viewer-image residual-mask" style={{ zIndex: 4, pointerEvents: 'none' }} draggable="false" onDragStart={(event) => event.preventDefault()} />
                      </div>
                    )}

                    {result && !loading && !filterLoading && !showDifference && (
                      <div className="image-stage" onMouseMove={handleViewerPointerMove}>
                        {activeFilterId && (
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

                        {!activeFilterId ? (
                          <img key={noisyImage} src={noisyImage} alt="Original" className="viewer-image single-image" draggable="false" onDragStart={(event) => event.preventDefault()} />
                        ) : (
                          <>
                            <img key={noisyImage} src={noisyImage} alt="Before" className="viewer-image base-image" draggable="false" onDragStart={(event) => event.preventDefault()} />
                            <img key={cleanedImage} src={cleanedImage} alt="After" className="viewer-image img-after" style={{ clipPath: `polygon(0 0, ${sliderPos}% 0, ${sliderPos}% 100%, 0 100%)` }} draggable="false" onDragStart={(event) => event.preventDefault()} />
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
                              ? (result.filter_metrics?.find(m => m.id === activeFilterId)?.name || 'Passthrough')
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
                          <th className="tooltip-trigger">
                            Execution Time
                            <span className="tooltip-content">Time in milliseconds to apply the filter. Lower is faster.</span>
                          </th>
                          <th className="tooltip-trigger">
                            Edge Preservation Ratio
                            <span className="tooltip-content">Correlation of image edge maps before and after filtering. 1.0 is perfect preservation.</span>
                          </th>
                          <th className="tooltip-trigger">
                            Variance of Laplacian (Sharpness)
                            <span className="tooltip-content">High-frequency detail variance of the restored image. Higher values represent a sharper image.</span>
                          </th>
                          <th className="tooltip-trigger">
                            BRISQUE
                            <span className="tooltip-content">Blind/Referenceless Image Spatial Quality Evaluator (0 to 100). Lower is better quality.</span>
                          </th>
                          <th className="tooltip-trigger">
                            NIQE
                            <span className="tooltip-content">Naturalness Image Quality Evaluator. Measures distance to natural image statistics. Lower is better.</span>
                          </th>
                          <th>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(result?.filter_metrics || []).map((filter) => {
                          const isActive = filter.id === activeFilterId;
                          return (
                            <tr
                              key={filter.id}
                              className={isActive ? 'row-selected' : ''}
                              onClick={() => handleFilterSwitch(filter.id)}
                              title={isActive ? 'Currently active filter' : `Apply ${filter.name}`}
                            >
                              <td style={{ fontWeight: isActive ? '600' : 'normal' }}>
                                {filter.name}
                              </td>
                              <td className="metric-value-mono">
                                {filter.runtime_ms !== null && filter.runtime_ms !== undefined
                                  ? `${filter.runtime_ms} ms`
                                  : '—'}
                              </td>
                              <td className="metric-value-mono">
                                {filter.edge_preservation !== null && filter.edge_preservation !== undefined
                                  ? filter.edge_preservation.toFixed(4)
                                  : '—'}
                              </td>
                              <td className="metric-value-mono">
                                {filter.laplacian_var !== null && filter.laplacian_var !== undefined
                                  ? filter.laplacian_var.toFixed(1)
                                  : '—'}
                              </td>
                              <td className="metric-value-mono">
                                {filter.brisque !== null && filter.brisque !== undefined
                                  ? filter.brisque.toFixed(2)
                                  : '—'}
                              </td>
                              <td className="metric-value-mono">
                                {filter.niqe !== null && filter.niqe !== undefined
                                  ? filter.niqe.toFixed(2)
                                  : '—'}
                              </td>
                              <td>
                                {isActive ? (
                                  result?.images?.all_cleaned?.[filter.id] ? (
                                    <span className="row-selected-badge">Active</span>
                                  ) : (
                                    <span className="row-caching-badge">Caching...</span>
                                  )
                                ) : backgroundLoadingIds.includes(filter.id) ? (
                                  <span className="row-caching-badge">Caching...</span>
                                ) : backgroundQueue.includes(filter.id) ? (
                                  <span className="row-queued-badge">Queued</span>
                                ) : (
                                  <span className="row-apply-badge">Apply</span>
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