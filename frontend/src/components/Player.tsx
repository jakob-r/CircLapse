import { useEffect, useMemo, useState } from "react";
import type { ProjectInfo } from "../api";
import { imageRoute } from "../routes";

/** Plays the aligned previews of all frames at the export frame rate. */
export function Player({
  project,
  onSkip,
  onClose,
}: {
  project: ProjectInfo;
  onSkip: (name: string) => void;
  onClose: () => void;
}) {
  const frames = useMemo(() => {
    const skipped = new Set(project.skipped);
    const byName = new Map(project.photos.map((p) => [p.name, p]));
    return project.order
      .filter((n) => !skipped.has(n))
      .map((n) => byName.get(n))
      .filter((p) => p !== undefined && p.usable && !p.dropped && p.preview !== null)
      .map((p) => ({ name: p!.name, src: p!.preview! }));
  }, [project]);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [loaded, setLoaded] = useState(0);
  const fps = project.settings.fps;
  const current = frames[Math.min(index, frames.length - 1)];

  // Preload all frames so playback does not stutter
  useEffect(() => {
    let done = 0;
    const els = frames.map((f) => {
      const el = new window.Image();
      el.onload = el.onerror = () => setLoaded(++done);
      el.src = f.src;
      return el;
    });
    return () => els.forEach((el) => (el.onload = el.onerror = null));
  }, [frames]);

  useEffect(() => {
    if (!playing || frames.length < 2) return;
    const t = window.setInterval(() => setIndex((i) => (i + 1) % frames.length), 1000 / fps);
    return () => window.clearInterval(t);
  }, [playing, fps, frames.length]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const step = (d: number) => {
        setPlaying(false);
        setIndex((i) => (i + d + frames.length) % frames.length);
      };
      const actions: Record<string, () => void> = {
        Escape: onClose,
        " ": () => setPlaying((p) => !p),
        ArrowLeft: () => step(-1),
        ArrowRight: () => step(1),
      };
      const action = actions[e.key];
      if (action) {
        e.preventDefault();
        action();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [frames.length, onClose]);

  return (
    <div className="overlay" onClick={onClose}>
      <div className="player" onClick={(e) => e.stopPropagation()}>
        {current ? (
          <img className="player-frame" src={current.src} alt={current.name} />
        ) : (
          <div className="player-frame message">No frames.</div>
        )}
        <input
          type="range"
          min={0}
          max={Math.max(0, frames.length - 1)}
          value={Math.min(index, frames.length - 1)}
          onChange={(e) => {
            setPlaying(false);
            setIndex(Number(e.target.value));
          }}
        />
        <div className="player-controls">
          <button onClick={() => setPlaying((p) => !p)}>{playing ? "❚❚ Pause" : "▶ Play"}</button>
          <span className="meta">
            {Math.min(index, frames.length - 1) + 1} / {frames.length} · {fps} fps
            {loaded < frames.length && ` · loading ${loaded}/${frames.length}`}
          </span>
          <span className="spacer" />
          {current && (
            <>
              <a href={imageRoute(current.name)} title={current.name}>
                Correct ellipse
              </a>
              <button
                onClick={() => {
                  setPlaying(false);
                  onSkip(current.name);
                }}
              >
                Remove frame
              </button>
            </>
          )}
          <button onClick={onClose}>
            Close <kbd>esc</kbd>
          </button>
        </div>
        <p className="meta">Low resolution preview; space plays and pauses, ← → step.</p>
      </div>
    </div>
  );
}
