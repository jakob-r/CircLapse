import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import {
  exportFileUrl,
  FORMATS,
  startExport,
  useExportStatus,
  type ExportSettings,
  type Format,
  type ProjectInfo,
  type ProjectUpdate,
} from "../api";

export function ExportDialog({
  project,
  update,
  onClose,
}: {
  project: ProjectInfo;
  update: (fn: (u: ProjectUpdate) => ProjectUpdate) => void;
  onClose: () => void;
}) {
  const queryClient = useQueryClient();
  const status = useExportStatus();
  const [settings, setSettings] = useState<ExportSettings>(project.settings);
  const [startError, setStartError] = useState<string | null>(null);
  const timer = useRef<number | undefined>(undefined);
  const running = status.data?.state === "running";

  // Save settings shortly after the last change, so frame count and crop update
  const change = (patch: Partial<ExportSettings>) => {
    const next = { ...settings, ...patch };
    setSettings(next);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => update((u) => ({ ...u, settings: next })), 300);
  };
  useEffect(() => () => window.clearTimeout(timer.current), []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const toggleFormat = (f: Format, on: boolean) =>
    change({
      formats: FORMATS.filter((x) => (x === f ? on : settings.formats.includes(x))),
    });

  const run = async () => {
    setStartError(null);
    try {
      queryClient.setQueryData(["export"], await startExport(settings));
    } catch (e) {
      setStartError(String(e));
    }
  };

  const s = status.data;
  return (
    <div className="overlay" onClick={onClose}>
      <div className="dialog" onClick={(e) => e.stopPropagation()}>
        <h2>Export video</h2>
        <div className="form">
          <span>Formats</span>
          <div className="formats">
            {FORMATS.map((f) => (
              <label key={f}>
                <input
                  type="checkbox"
                  checked={settings.formats.includes(f)}
                  onChange={(e) => toggleFormat(f, e.target.checked)}
                />
                {f}
              </label>
            ))}
          </div>
          <label htmlFor="size">Size</label>
          <select
            id="size"
            value={settings.size}
            onChange={(e) => change({ size: Number(e.target.value) })}
          >
            {[540, 720, 1080, 1440, 2160].map((v) => (
              <option key={v} value={v}>
                {v} × {v}
              </option>
            ))}
          </select>
          <label htmlFor="fps">Speed</label>
          <div className="range">
            <input
              id="fps"
              type="range"
              min={1}
              max={30}
              step={1}
              value={settings.fps}
              onChange={(e) => change({ fps: Number(e.target.value) })}
            />
            <span>{settings.fps} fps</span>
          </div>
          <label htmlFor="keep" title="Dropping photos whose circle is near the border widens the crop">
            Keep photos
          </label>
          <div className="range">
            <input
              id="keep"
              type="range"
              min={0.5}
              max={1}
              step={0.05}
              value={settings.keep}
              onChange={(e) => change({ keep: Number(e.target.value) })}
            />
            <span>{Math.round(settings.keep * 100)}%</span>
          </div>
          <label htmlFor="zoom" title="Crop tighter than the widest crop that fits all kept photos">
            Zoom in
          </label>
          <div className="range">
            <input
              id="zoom"
              type="range"
              min={1}
              max={3}
              step={0.05}
              value={settings.zoom}
              onChange={(e) => change({ zoom: Number(e.target.value) })}
            />
            <span>{settings.zoom === 1 ? "none" : `${settings.zoom.toFixed(2)}×`}</span>
          </div>
          <span>Color</span>
          <label>
            <input
              type="checkbox"
              checked={settings.grade}
              onChange={(e) => change({ grade: e.target.checked })}
            />
            Automatic color and exposure grade
          </label>
        </div>
        <p className="meta">
          {project.frames} frames, {(project.frames / settings.fps).toFixed(1)} s
          {project.extent !== null &&
            ` · circle fills ${Math.round(100 / project.extent)}% of the frame`}
        </p>

        {s && s.state !== "idle" && (
          <div className={`export-status ${s.state}`}>
            {s.state === "running" && <div>Rendering… {s.log.at(-1)}</div>}
            {s.state === "error" && <div>Export failed: {s.error}</div>}
            {s.state === "done" && (
              <>
                <div>
                  Done. Saved in <code>{s.directory}</code>
                </div>
                <ul>
                  {s.outputs.map((o) => (
                    <li key={o}>
                      <a href={exportFileUrl(o)} download={o}>
                        {o}
                      </a>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>
        )}
        {startError && <div className="export-status error">{startError}</div>}

        <div className="dialog-buttons">
          <button onClick={onClose}>Close</button>
          <button
            className="primary"
            disabled={running || settings.formats.length === 0 || project.frames === 0}
            onClick={() => void run()}
          >
            {running ? "Rendering…" : "Export"}
          </button>
        </div>
      </div>
    </div>
  );
}
