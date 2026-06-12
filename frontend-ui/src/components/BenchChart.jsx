import React, { useState } from 'react';
import { FILTER_COLORS, FIXED_FILTER_ORDER, SHORT_NAME, formatRuntime } from '../utils/helpers';

export default function BenchChart({ data, filterNames, metric }) {
  const [hoveredDot, setHoveredDot] = useState(null);

  if (!data?.length) return null;
  const allFids = Object.keys(data[0]?.results || {});
  if (!allFids.length) return null;

  const fids = [
    ...FIXED_FILTER_ORDER.filter(fid => allFids.includes(fid)),
    ...allFids.filter(fid => !FIXED_FILTER_ORDER.includes(fid))
  ];

  const width = 480;
  const height = 370;
  const paddingLeft = 25;
  const paddingRight = 15;
  const paddingTop = 30;
  const paddingBottom = 35;

  const chartWidth = width - paddingLeft - paddingRight;
  const chartHeight = height - paddingTop - paddingBottom;

  const allValues = [];
  data.forEach(lvl => {
    fids.forEach(fid => {
      const v = lvl.results[fid]?.[metric];
      if (v != null) allValues.push(v);
    });
  });

  let minVal = Math.min(...allValues);
  let maxVal = Math.max(...allValues);

  if (allValues.length === 0) {
    minVal = 0;
    maxVal = 100;
  } else if (minVal === maxVal) {
    minVal = minVal * 0.9;
    maxVal = maxVal * 1.1;
  } else {
    const range = maxVal - minVal;
    if (metric === 'time_ms') {
      minVal = 0;
      maxVal = maxVal + range * 0.15;
    } else if (metric === 'ssim') {
      minVal = Math.max(0.0, minVal - range * 0.15);
      maxVal = Math.min(1.0, maxVal + range * 0.15);
    } else {
      minVal = Math.max(0.0, minVal - range * 0.15);
      maxVal = maxVal + range * 0.15;
    }
  }

  const ticksCount = 5;
  const yTicks = [];
  for (let i = 0; i < ticksCount; i++) {
    yTicks.push(minVal + (i * (maxVal - minVal)) / (ticksCount - 1));
  }

  const getX = (index) => {
    return paddingLeft + (index / (data.length - 1)) * chartWidth;
  };

  const getY = (val) => {
    if (val == null) return paddingTop + chartHeight;
    const ratio = (val - minVal) / (maxVal - minVal);
    return paddingTop + chartHeight - ratio * chartHeight;
  };

  const formatValue = (val) => {
    if (val == null) return '—';
    if (metric === 'psnr') {
      return `${val.toFixed(1)} dB`;
    }
    if (metric === 'ssim') {
      return val.toFixed(3);
    }
    if (metric === 'time_ms') {
      return formatRuntime(val);
    }
    return val;
  };

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ flex: 1, position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <svg viewBox={`0 0 ${width} ${height}`} className="bench-chart-svg" preserveAspectRatio="xMidYMid meet">
          {yTicks.map((tick, i) => {
            const y = getY(tick);
            return (
              <g key={`grid-${i}`}>
                <line
                  x1={paddingLeft}
                  y1={y}
                  x2={width - paddingRight}
                  y2={y}
                  className="bench-chart-grid-line"
                />
                <text
                  x={paddingLeft - 8}
                  y={y + 3}
                  textAnchor="end"
                  className="bench-chart-text"
                >
                  {metric === 'ssim' ? tick.toFixed(2) : Math.round(tick)}
                </text>
              </g>
            );
          })}

          {data.map((lvl, index) => {
            const x = getX(index);
            return (
              <text
                key={`label-x-${index}`}
                x={x}
                y={height - paddingBottom + 16}
                textAnchor="middle"
                className="bench-chart-text"
              >
                {lvl.label}
              </text>
            );
          })}

          {fids.map(fid => {
            const points = data.map((lvl, index) => {
              const val = lvl.results[fid]?.[metric];
              return val != null ? `${getX(index)},${getY(val)}` : null;
            }).filter(p => p != null).join(' ');

            return (
              <polyline
                key={`line-${fid}`}
                points={points}
                className="bench-chart-line"
                style={{ stroke: FILTER_COLORS[fid] }}
              />
            );
          })}

          {fids.map(fid =>
            data.map((lvl, index) => {
              const val = lvl.results[fid]?.[metric];
              if (val == null) return null;
              const cx = getX(index);
              const cy = getY(val);
              return (
                <g key={`dot-${fid}-${index}`}>
                  <circle
                    cx={cx}
                    cy={cy}
                    r={3.5}
                    className="bench-chart-dot-bg"
                  />
                  <circle
                    cx={cx}
                    cy={cy}
                    r={3}
                    className="bench-chart-dot"
                    style={{ stroke: FILTER_COLORS[fid], fill: FILTER_COLORS[fid] }}
                    onMouseEnter={(e) => {
                      const rect = e.currentTarget.getBoundingClientRect();
                      const containerRect = e.currentTarget.ownerSVGElement.parentNode.getBoundingClientRect();
                      
                      const overlapping = [];
                      const yVal = getY(val);
                      fids.forEach(otherFid => {
                        const otherVal = lvl.results[otherFid]?.[metric];
                        if (otherVal != null) {
                          const yOther = getY(otherVal);
                          if (Math.abs(yOther - yVal) <= 3) {
                            overlapping.push({
                              filterName: SHORT_NAME(filterNames?.[otherFid]),
                              color: FILTER_COLORS[otherFid],
                              val: otherVal,
                              y: yOther
                            });
                          }
                        }
                      });

                      overlapping.sort((a, b) => a.y - b.y);

                      setHoveredDot({
                        x: rect.left - containerRect.left + rect.width / 2,
                        y: rect.top - containerRect.top,
                        levelLabel: lvl.label,
                        filters: overlapping
                      });
                    }}
                    onMouseLeave={() => setHoveredDot(null)}
                  />
                </g>
              );
            })
          )}

          <line
            x1={paddingLeft}
            y1={paddingTop}
            x2={paddingLeft}
            y2={height - paddingBottom}
            className="bench-chart-axis-line"
          />
          <line
            x1={paddingLeft}
            y1={height - paddingBottom}
            x2={width - paddingRight}
            y2={height - paddingBottom}
            className="bench-chart-axis-line"
          />
        </svg>

        {hoveredDot && (
          <span
            className="tooltip-content chart-tooltip-content"
            style={{
              visibility: 'visible',
              opacity: 1,
              position: 'absolute',
              left: `${hoveredDot.x}px`,
              top: `${hoveredDot.y - 8}px`,
              transform: 'translate(-50%, -100%)',
              pointerEvents: 'none',
              textAlign: 'left',
              zIndex: 1000,
              background: 'var(--bg-panel)',
              border: '1px solid var(--border-color)',
              boxShadow: '0 10px 28px -8px rgba(0, 0, 0, 0.5)',
              padding: '8px 12px',
              borderRadius: '8px',
              fontSize: '11px',
              color: 'var(--text-main)',
              width: 'max-content',
              display: 'flex',
              flexDirection: 'column',
              gap: '4px'
            }}
          >
            <span style={{ fontWeight: 700, color: 'var(--text-header)', display: 'block', borderBottom: '1px solid var(--border-color)', paddingBottom: '3px', marginBottom: '2px' }}>
              {hoveredDot.levelLabel}
            </span>
            {hoveredDot.filters.map((f, fi) => (
              <span key={fi} style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: f.color, display: 'inline-block', flexShrink: 0 }} />
                <strong>{f.filterName}</strong>: {formatValue(f.val)}
              </span>
            ))}
          </span>
        )}
      </div>
    </div>
  );
}
