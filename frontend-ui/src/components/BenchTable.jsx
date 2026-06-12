import React from 'react';
import { FILTER_COLORS, FIXED_FILTER_ORDER, SHORT_NAME, formatRuntime } from '../utils/helpers';

export default function BenchTable({ data, filterNames, metric }) {
  if (!data?.length) return null;
  const allFids = Object.keys(data[0]?.results || {});
  if (!allFids.length) return null;

  const fids = [
    ...FIXED_FILTER_ORDER.filter(fid => allFids.includes(fid)),
    ...allFids.filter(fid => !FIXED_FILTER_ORDER.includes(fid))
  ];

  const avgOf = (fid) => {
    const vals = data.map(l => l.results[fid]?.[metric]).filter(v => v != null);
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  };

  const rows = fids.map(fid => {
    const avg = avgOf(fid);
    return {
      fid,
      avg,
      name: SHORT_NAME(filterNames?.[fid])
    };
  });

  const bestValuePerLevel = data.map((lvl, index) => {
    let bestFid = null;
    let bestVal = metric === 'time_ms' ? Infinity : -Infinity;

    fids.forEach(fid => {
      const val = lvl.results[fid]?.[metric];
      if (val != null) {
        if (metric === 'time_ms') {
          if (val < bestVal) {
            bestVal = val;
            bestFid = fid;
          }
        } else {
          if (val > bestVal) {
            bestVal = val;
            bestFid = fid;
          }
        }
      }
    });
    return bestFid;
  });

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
    <div className="bench-table-wrapper">
      <table className="bench-table">
        <thead>
          <tr>
            <th>Filter</th>
            {data.map(l => <th key={l.label}>{l.label}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const { fid } = row;
            return (
              <tr key={fid}>
                <td className="bench-filter-cell">
                  <span className="eval-filter-dot" style={{ background: FILTER_COLORS[fid] }} />
                  {SHORT_NAME(filterNames?.[fid])}
                </td>
                {data.map((lvl, li) => {
                  const val = lvl.results[fid]?.[metric];
                  const isBest = fid === bestValuePerLevel[li];
                  return (
                    <td key={lvl.label} className={`bench-cell${isBest ? ' bench-cell--best' : ''}`}>
                      <span className="bench-psnr">
                        {formatValue(val)}
                      </span>
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
