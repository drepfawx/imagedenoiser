import React from 'react';

export function Icon({ children }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false" className="viewer-icon">
      {children}
    </svg>
  );
}

export function NoiseIcon({ active }) {
  return (
    <Icon>
      <path d="M5 14l2-3 2 2 3-6 2 4 2-2 3 5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M4 19h16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      <path d={active ? 'M6 6l12 12' : 'M15.5 7.5a4 4 0 11-5.7 5.7A4 4 0 0115.5 7.5z'} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </Icon>
  );
}

export function MagnifierIcon() {
  return (
    <Icon>
      <circle cx="10" cy="10" r="5.5" fill="none" stroke="currentColor" strokeWidth="1.8" />
      <path d="M13.9 13.9L19 19" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      <path d="M8.2 10h3.6M10 8.2v3.6" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </Icon>
  );
}

export function SuccessShieldIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ width: '16px', height: '16px', color: 'var(--accent-green)' }}>
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" strokeLinecap="round" strokeLinejoin="round" />
      <path d="m9 11 2 2 4-4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function GaussianWavesIcon() {
  return (
    <svg className="animate-pulse" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ width: '16px', height: '16px', color: 'var(--accent-blue)' }}>
      <path d="M2 10s3-4 6-4 4 8 8 8 6-4 6-4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M2 14s3-4 6-4 4 8 8 8 6-4 6-4" opacity="0.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function SpecklesIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ width: '16px', height: '16px', color: 'var(--accent-yellow)' }}>
      <circle cx="8" cy="8" r="1.5" fill="currentColor" stroke="none" />
      <circle cx="16" cy="14" r="1.2" fill="currentColor" stroke="none" />
      <circle cx="7" cy="16" r="1" fill="currentColor" stroke="none" />
      <circle cx="18" cy="7" r="1.8" fill="currentColor" stroke="none" />
      <circle cx="12" cy="11" r="1.2" fill="currentColor" stroke="none" />
      <path d="M4 12h1M20 12h1M12 4v1M12 20v1" strokeLinecap="round" />
    </svg>
  );
}

export function PulseHeartbeatIcon() {
  return (
    <svg className="pulse-animation" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ width: '28px', height: '28px', color: 'var(--text-muted)' }}>
      <path d="M22 12h-4l-3 9L9 3l-3 9H2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function MonitorDisplayIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false" className="viewer-empty-icon">
      <rect x="4.5" y="5.5" width="15" height="10" rx="1.8" fill="none" stroke="currentColor" strokeWidth="1.5" />
      <path d="M9 19h6M12 15.5v3.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M8 10.2h1.8M14.2 10.2H16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M10.1 12.6h3.8" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}
