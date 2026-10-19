import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState, type FormEvent } from "react";
import {
  dismissImport,
  nativeApi,
  openProject,
  startImport,
  useImportStatus,
  type ImportRequest,
  type ImportStatus,
  type LauncherInfo,
} from "../api";

const isoDay = (d: Date) =>
  `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;

/** True once the native window's bridge is there (it is injected after load). */
function useNative(): boolean {
  const [native, setNative] = useState(() => nativeApi() !== null);
  useEffect(() => {
    if (native) return;
    const ready = () => setNative(nativeApi() !== null);
    window.addEventListener("pywebviewready", ready);
    return () => window.removeEventListener("pywebviewready", ready);
  }, [native]);
  return native;
}

/** Whether files from outside the page are dragged over the window. */
function useFileDrag(): boolean {
  const [over, setOver] = useState(false);
  const depth = useRef(0);
  useEffect(() => {
    const isFiles = (e: DragEvent) => e.dataTransfer?.types.includes("Files") ?? false;
    const enter = (e: DragEvent) => {
      if (!isFiles(e)) return;
      depth.current += 1;
      setOver(true);
    };
    const leave = (e: DragEvent) => {
      if (!isFiles(e)) return;
      depth.current = Math.max(0, depth.current - 1);
      if (depth.current === 0) setOver(false);
    };
    const drop = () => {
      depth.current = 0;
      setOver(false);
    };
    window.addEventListener("dragenter", enter);
    window.addEventListener("dragleave", leave);
    window.addEventListener("drop", drop);
    return () => {
      window.removeEventListener("dragenter", enter);
      window.removeEventListener("dragleave", leave);
      window.removeEventListener("drop", drop);
    };
  }, []);
  return over;
}

function Progress({ status, onBack }: { status: ImportStatus; onBack: () => void }) {
  const failed = status.state === "error";
  const fraction = status.total > 0 ? status.done / status.total : null;
  const log = status.log ?? [];
  return (
    <div className="start-progress">
      <h2>{failed ? "Import failed" : status.step || "Importing"}</h2>
      {failed ? (
        <p className="message error">{status.error}</p>
      ) : (
        <>
          <div className={`progress ${fraction === null ? "indeterminate" : ""}`}>
            <div style={{ width: fraction === null ? undefined : `${fraction * 100}%` }} />
          </div>
          <span className="meta">
            {fraction !== null ? `${status.done} of ${status.total} photos` : " "}
          </span>
        </>
      )}
      <pre className="log">{log.slice(-8).join("\n")}</pre>
      {failed && (
        <div>
          <button onClick={onBack}>Back</button>
        </div>
      )}
    </div>
  );
}

export function Start({ launcher }: { launcher: LauncherInfo }) {
  const queryClient = useQueryClient();
  const status = useImportStatus();
  const native = useNative();
  const dragging = useFileDrag();
  const [error, setError] = useState<string | null>(null);
  const [path, setPath] = useState("");
  const [confidence, setConfidence] = useState(launcher.min_confidence);
  // The last seven days
  const [fromDate, setFromDate] = useState(() => isoDay(new Date(Date.now() - 6 * 864e5)));
  const [toDate, setToDate] = useState(() => isoDay(new Date()));

  useEffect(() => {
    document.title = "circlapse";
  }, []);

  const state = status.data?.state;
  useEffect(() => {
    if (state !== "done") return;
    // The server opened the new project: show its studio
    void dismissImport().then((s) => {
      queryClient.setQueryData(["import"], s);
      void queryClient.invalidateQueries({ queryKey: ["launcher"] });
      void queryClient.invalidateQueries({ queryKey: ["project"] });
      void queryClient.invalidateQueries({ queryKey: ["images"] });
    });
  }, [state, queryClient]);

  const run = async (request: ImportRequest) => {
    setError(null);
    try {
      const withConfidence = { ...request, min_confidence: confidence };
      queryClient.setQueryData(["import"], await startImport(withConfidence));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };
  const importPaths = (paths: string[]) => void run({ source: "paths", paths });

  const latest = useRef(importPaths);
  useEffect(() => {
    latest.current = importPaths;
  });
  useEffect(() => {
    const onDrop = (e: Event) => latest.current((e as CustomEvent<string[]>).detail);
    window.addEventListener("circlapse:drop", onDrop);
    return () => window.removeEventListener("circlapse:drop", onDrop);
  }, []);

  const pick = async () => {
    const folder = await nativeApi()?.pick_folder();
    if (folder) importPaths([folder]);
  };

  const open = async (p: string) => {
    setError(null);
    try {
      queryClient.setQueryData(["launcher"], await openProject(p));
      void queryClient.invalidateQueries({ queryKey: ["project"] });
      void queryClient.invalidateQueries({ queryKey: ["images"] });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const submitPath = (e: FormEvent) => {
    e.preventDefault();
    if (path.trim()) importPaths([path.trim()]);
  };
  const submitPhotos = (e: FormEvent) => {
    e.preventDefault();
    void run({ source: "apple_photos", paths: [], from_date: fromDate, to_date: toDate });
  };

  const s = status.data;
  const busy = s?.state === "running" || s?.state === "done";
  if (s && (busy || s.state === "error")) {
    const back = () => void dismissImport().then((d) => queryClient.setQueryData(["import"], d));
    return (
      <div className="start">
        <Progress status={s} onBack={back} />
      </div>
    );
  }

  return (
    <div className="start">
      <h1>circlapse</h1>
      <p className="meta">Find the circles in your photos and line them up into a hyperlapse.</p>
      {error && <p className="message error">{error}</p>}

      <div className="start-sources">
        <section className={`start-card drop-zone ${dragging && native ? "over" : ""}`}>
          <h2>From a folder</h2>
          {native ? (
            <>
              <p className="meta">Drop a folder or photos anywhere in this window, or</p>
              <div>
                <button className="primary" onClick={() => void pick()}>
                  Choose folder…
                </button>
              </div>
            </>
          ) : (
            <form onSubmit={submitPath}>
              <p className="meta">Path of a folder with photos</p>
              <div className="start-row">
                <input
                  value={path}
                  onChange={(e) => setPath(e.target.value)}
                  placeholder="~/Pictures/circles"
                  spellCheck={false}
                />
                <button className="primary" type="submit" disabled={!path.trim()}>
                  Import
                </button>
              </div>
            </form>
          )}
          <p className="hint">
            The project goes into a <code>circlapse</code> folder next to the photos; the photos
            themselves stay untouched.
          </p>
        </section>

        <section className="start-card">
          <h2>From Apple Photos</h2>
          {launcher.apple_photos ? (
            <form onSubmit={submitPhotos}>
              <div className="form">
                <label htmlFor="from">First day</label>
                <input
                  id="from"
                  type="date"
                  value={fromDate}
                  max={toDate}
                  onChange={(e) => setFromDate(e.target.value)}
                />
                <label htmlFor="to">Last day</label>
                <input
                  id="to"
                  type="date"
                  value={toDate}
                  min={fromDate}
                  onChange={(e) => setToDate(e.target.value)}
                />
              </div>
              <div className="start-row">
                <button className="primary" type="submit" disabled={!fromDate || !toDate}>
                  Import
                </button>
              </div>
              <p className="hint">
                Exported to <code>{launcher.library}</code>. Needs Full Disk Access for the
                terminal that started the studio.
              </p>
            </form>
          ) : (
            <p className="hint">
              Install the photos extra (<code>uv sync --extra photos</code>) to import from Apple
              Photos.
            </p>
          )}
        </section>
      </div>

      <section className="start-threshold">
        <label
          htmlFor="confidence"
          title="Photos whose circle the detector is less sure about than this are not imported"
        >
          Minimum confidence
        </label>
        <div className="range">
          <input
            id="confidence"
            type="range"
            min={0}
            max={0.99}
            step={0.01}
            value={confidence}
            onChange={(e) => setConfidence(Number(e.target.value))}
          />
          <span>{confidence.toFixed(2)}</span>
        </div>
        <p className="hint">Lower keeps more photos, but more of them without a real circle.</p>
      </section>

      {launcher.recent.length > 0 && (
        <section className="start-recent">
          <h2>Recent projects</h2>
          <ul>
            {launcher.recent.map((r) => (
              <li key={r.path}>
                <button onClick={() => void open(r.path)} title={r.path}>
                  <span className="name">{r.name}</span>
                  <span className="meta">
                    {r.photos} photos · {r.path}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      {native && dragging && <div className="drop-overlay">Drop to make a circlapse</div>}
    </div>
  );
}
