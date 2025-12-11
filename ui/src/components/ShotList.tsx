import { useState } from 'react';
import type { Shot } from '../types/shot';
import './ShotList.css';

const SHOTS_PER_PAGE = 5;

interface ShotListProps {
  shots: Shot[];
}

export function ShotList({ shots }: ShotListProps) {
  const [page, setPage] = useState(0);

  const totalPages = Math.ceil(shots.length / SHOTS_PER_PAGE);
  const reversedShots = [...shots].reverse();
  const startIndex = page * SHOTS_PER_PAGE;
  const pageShots = reversedShots.slice(startIndex, startIndex + SHOTS_PER_PAGE);

  if (shots.length === 0) {
    return (
      <div className="shot-list shot-list--empty">
        <p>No shots recorded yet</p>
      </div>
    );
  }

  return (
    <div className="shot-list">
      <div className="shot-list__rows">
        {pageShots.map((shot, index) => {
          const shotNumber = shots.length - startIndex - index;
          return (
            <div key={shot.timestamp} className="shot-row">
              <span className="shot-row__number">#{shotNumber}</span>
              <span className="shot-row__club">{shot.club}</span>
              <span className="shot-row__speed">{shot.ball_speed_mph.toFixed(1)}</span>
              <span className="shot-row__carry">{shot.estimated_carry_yards} yds</span>
              {shot.smash_factor && (
                <span className="shot-row__smash">{shot.smash_factor.toFixed(2)}</span>
              )}
            </div>
          );
        })}
      </div>

      {totalPages > 1 && (
        <div className="pagination">
          <button
            className="pagination__button"
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            disabled={page === 0}
          >
            Prev
          </button>
          <span className="pagination__info">
            {page + 1} / {totalPages}
          </span>
          <button
            className="pagination__button"
            onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
            disabled={page === totalPages - 1}
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
