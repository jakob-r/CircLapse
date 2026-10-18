import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  fetchLabel,
  fitEllipse,
  imageUrl,
  saveLabel,
  useImages,
  usePrediction,
  useProject,
  type ImageInfo,
  type Label,
} from "../api";
import { cropCorners, maxExtent, type Ellipse, type Point } from "../geometry";
import { addEllipse, deleteEllipse, setProminent, toggleStatus, updateEllipse } from "../labelOps";
import { gridRoute, imageRoute } from "../routes";
import { EllipseCanvas } from "./EllipseCanvas";
import { storedView } from "./Studio";

type SaveState = "saved" | "pending" | "saving" | "error";

function useImageElement(src: string) {
  const [img, setImg] = useState<HTMLImageElement>();
  useEffect(() => {
    const el = new window.Image();
    el.onload = () => setImg(el);
    el.src = src;
    return () => {
      el.onload = null;
    };
  }, [src]);
  return img;
}

/** Debounced autosave; flushes pending changes on unmount (e.g. navigation). */
function useAutosave(name: string, onSaved: () => void) {
  const [state, setState] = useState<SaveState>("saved");
  const pending = useRef<Label | null>(null);
  const timer = useRef<number | undefined>(undefined);

  const flush = useCallback(async () => {
    window.clearTimeout(timer.current);
    const l = pending.current;
    if (!l) return;
    pending.current = null;
    setState("saving");
    try {
      await saveLabel(name, l);
      setState(pending.current ? "pending" : "saved");
      onSaved();
    } catch {
      setState("error");
    }
  }, [name, onSaved]);

  const schedule = useCallback(
    (l: Label) => {
      pending.current = l;
      setState("pending");
      window.clearTimeout(timer.current);
      timer.current = window.setTimeout(flush, 400);
    },
    [flush],
  );

  useEffect(() => () => void flush(), [flush]);

  return { state, flush, schedule };
}

export function Editor({ name }: { name: string }) {
  const images = useImages();
  const labelQuery = useQuery({
    queryKey: ["label", name],
    queryFn: () => fetchLabel(name),
    staleTime: Infinity,
    gcTime: 0,
  });
  const info = images.data?.find((i) => i.name === name);

  if (images.error || labelQuery.error)
    return <div className="message error">{String(images.error ?? labelQuery.error)}</div>;
  if (!images.data || !labelQuery.data) return <div className="message">Loading…</div>;
  if (!info) return <div className="message error">Unknown image {name}</div>;
  return <EditorLoaded info={info} images={images.data} initial={labelQuery.data} />;
}

function EditorLoaded({
  info,
  images,
  initial,
}: {
  info: ImageInfo;
  images: ImageInfo[];
  initial: Label;
}) {
  const { name } = info;
  const queryClient = useQueryClient();
  const image = useImageElement(imageUrl(name));
  const prediction = usePrediction(name).data;

  const [label, setLabel] = useState<Label>(initial);
  const latest = useRef(label);
  const [selected, setSelected] = useState<number | null>(initial.prominent);
  const [drawing, setDrawing] = useState(false);
  const [points, setPoints] = useState<Point[]>([]);
  const [preview, setPreview] = useState<Ellipse | null>(null);
  const [fitSignal, setFitSignal] = useState(0);

  const onSaved = useCallback(
    () => void queryClient.invalidateQueries({ queryKey: ["images"] }),
    [queryClient],
  );
  const save = useAutosave(name, onSaved);
  const update = (fn: (l: Label) => Label) => {
    const next = fn(latest.current);
    latest.current = next;
    setLabel(next);
    save.schedule(next);
  };

  const index = images.findIndex((i) => i.name === name);
  const go = (delta: number) => {
    const next = images[index + delta];
    if (next) window.location.hash = imageRoute(next.name);
  };

  // Coming from the studio's aligned view: dim what the video crops away, so the photos
  // that limit the common crop stand out
  const project = useProject().data;
  const prominentEllipse = label.prominent !== null ? label.ellipses[label.prominent] : undefined;
  const extent = project?.extent ?? null;
  const showCrop = storedView() === "aligned" && prominentEllipse && extent !== null;
  const crop = showCrop ? cropCorners(prominentEllipse, extent) : undefined;
  const room = showCrop ? maxExtent(prominentEllipse, info.width, info.height) / extent : null;

  const startDrawing = () => {
    setDrawing(true);
    setPoints([]);
    setPreview(null);
  };
  const stopDrawing = () => {
    setDrawing(false);
    setPoints([]);
    setPreview(null);
  };
  const addPoint = async (p: Point) => {
    const pts = [...points, p];
    setPoints(pts);
    if (pts.length >= 5) setPreview(await fitEllipse(pts));
  };
  const commitDrawing = () => {
    if (!preview) return;
    const n = label.ellipses.length;
    update((l) => addEllipse(l, preview));
    setSelected(n);
    stopDrawing();
  };
  const accept = (e: Ellipse) => {
    const n = label.ellipses.length;
    update((l) => addEllipse(l, e));
    setSelected(n);
  };
  const remove = (i: number) => {
    update((l) => deleteEllipse(l, i));
    setSelected(null);
  };

  const predictions = prediction?.ellipses ?? [];

  // keyboard shortcuts; re-registered on every render so handlers see current state
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;
      if ((e.metaKey || e.ctrlKey) && e.key === "s") {
        e.preventDefault();
        void save.flush();
        return;
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const actions: Record<string, () => void> = {
        ArrowLeft: () => go(-1),
        ArrowRight: () => go(1),
        Escape: () => (drawing ? stopDrawing() : setSelected(null)),
        Enter: commitDrawing,
        d: () => (drawing ? stopDrawing() : startDrawing()),
        f: () => setFitSignal((s) => s + 1),
        p: () => selected !== null && update((l) => setProminent(l, selected)),
        n: () => update((l) => toggleStatus(l, "no_ellipse")),
        s: () => update((l) => toggleStatus(l, "skip")),
        a: () => predictions[0] && accept(predictions[0]),
        Delete: () => selected !== null && remove(selected),
        Backspace: () => selected !== null && remove(selected),
        g: () => (window.location.hash = gridRoute),
      };
      const action = actions[e.key];
      if (action) {
        e.preventDefault();
        action();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  return (
    <div className="editor">
      <EllipseCanvas
        image={image}
        width={info.width}
        height={info.height}
        ellipses={label.ellipses}
        prominent={label.prominent}
        predictions={predictions}
        selected={selected}
        drawing={drawing}
        drawPoints={points}
        drawPreview={preview}
        fitSignal={fitSignal}
        crop={crop}
        onSelect={setSelected}
        onChange={(i, e) => update((l) => updateEllipse(l, i, e))}
        onAcceptPrediction={accept}
        onAddPoint={(p) => void addPoint(p)}
      />
      <aside className="panel">
        <div className="nav-row">
          <a href={gridRoute}>← Grid</a>
          <span>
            {index + 1} / {images.length}
          </span>
          <span>
            <button disabled={index <= 0} onClick={() => go(-1)}>
              ◀
            </button>
            <button disabled={index >= images.length - 1} onClick={() => go(1)}>
              ▶
            </button>
          </span>
        </div>
        <h2 title={name}>{name}</h2>
        <div className="meta">
          {info.width}×{info.height} · <span className={`save ${save.state}`}>{save.state}</span>
        </div>
        {room !== null && (
          <div className="meta" title="How far the video crop could grow before leaving this photo">
            {room < 1 - 1e-9
              ? "Cropped out: the video frame leaves this photo"
              : room < 1.02
                ? "Limits the crop: the video frame touches the border"
                : `Crop could grow ${Math.round((room - 1) * 100)}% in this photo`}
          </div>
        )}

        <section>
          <h3>Status</h3>
          <div className="status-row">
            <span className={`badge ${label.status}`}>{label.status.replace("_", " ")}</span>
            <button
              className={label.status === "no_ellipse" ? "active" : ""}
              onClick={() => update((l) => toggleStatus(l, "no_ellipse"))}
            >
              No ellipse <kbd>n</kbd>
            </button>
            <button
              className={label.status === "skip" ? "active" : ""}
              onClick={() => update((l) => toggleStatus(l, "skip"))}
            >
              Skip <kbd>s</kbd>
            </button>
          </div>
        </section>

        <section>
          <h3>Ellipses</h3>
          {drawing ? (
            <div className="drawing-hint">
              Click points on the rim ({points.length} placed
              {points.length < 5 ? `, need ${5 - points.length} more` : ""}).
              <div>
                <button disabled={!preview} onClick={commitDrawing}>
                  Add <kbd>⏎</kbd>
                </button>
                <button onClick={stopDrawing}>
                  Cancel <kbd>esc</kbd>
                </button>
              </div>
            </div>
          ) : (
            <button onClick={startDrawing}>
              + Draw ellipse <kbd>d</kbd>
            </button>
          )}
          <ul className="ellipse-list">
            {label.ellipses.map((e, i) => (
              <li
                key={i}
                className={i === selected ? "selected" : ""}
                onClick={() => setSelected(i)}
              >
                <label title="Most prominent">
                  <input
                    type="radio"
                    checked={label.prominent === i}
                    onChange={() => update((l) => setProminent(l, i))}
                  />
                  #{i + 1}
                </label>
                <span className="dims">
                  ⌀ {Math.round(2 * Math.max(e.a, e.b))} · {Math.round(e.angle_deg)}°
                </span>
                <button
                  className="icon"
                  onClick={(evt) => {
                    evt.stopPropagation();
                    remove(i);
                  }}
                >
                  ✕
                </button>
              </li>
            ))}
          </ul>
        </section>

        {predictions.length > 0 && (
          <section>
            <h3>Predictions</h3>
            <ul className="ellipse-list">
              {predictions.map((e, i) => (
                <li key={i}>
                  <span>
                    #{i + 1}
                    {e.score != null && ` · ${e.score.toFixed(2)}`}
                  </span>
                  <button onClick={() => accept(e)}>Accept{i === 0 && <kbd>a</kbd>}</button>
                </li>
              ))}
            </ul>
          </section>
        )}

        <section className="help">
          <h3>Keys</h3>
          <dl>
            <dt>← →</dt>
            <dd>previous / next image</dd>
            <dt>d</dt>
            <dd>draw: click ≥5 rim points, ⏎ to add</dd>
            <dt>p</dt>
            <dd>mark selected as prominent</dd>
            <dt>⌫</dt>
            <dd>delete selected</dd>
            <dt>f</dt>
            <dd>fit view (wheel zooms, drag pans)</dd>
            <dt>g</dt>
            <dd>back to grid</dd>
          </dl>
          <p>
            Drag the selected ellipse to move it; the green handle sets size and rotation, the
            orange handle the other axis.
          </p>
        </section>
      </aside>
    </div>
  );
}
