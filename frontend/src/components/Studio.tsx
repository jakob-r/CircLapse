import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState, type DragEvent } from "react";
import {
  saveProject,
  useImages,
  useProject,
  type ImageInfo,
  type PhotoInfo,
  type ProjectInfo,
  type ProjectUpdate,
} from "../api";
import { moveItem, setSkipped } from "../projectOps";
import { imageRoute } from "../routes";
import { ExportDialog } from "./ExportDialog";
import { Player } from "./Player";
import { SortDialog } from "./SortDialog";
import { ThumbImage } from "./ThumbImage";

type View = "photos" | "aligned";
const VIEW_KEY = "circlapse.studio.view";

/** The grid view last chosen in the studio; the editor shows the crop when it is "aligned". */
export function storedView(): View {
  try {
    return sessionStorage.getItem(VIEW_KEY) === "aligned" ? "aligned" : "photos";
  } catch {
    return "photos";
  }
}

/** The grid view survives trips to the editor (which unmount the studio). */
function useView(): [View, (v: View) => void] {
  const [view, setView] = useState<View>(storedView);
  const set = (v: View) => {
    setView(v);
    try {
      sessionStorage.setItem(VIEW_KEY, v);
    } catch {
      // storage blocked: the view just resets
    }
  };
  return [view, set];
}

/** Optimistic project edits: the cache updates at once, PUTs run one after another and the
 * server's answer to the last one wins. */
export function useProjectUpdate() {
  const queryClient = useQueryClient();
  const chain = useRef(Promise.resolve());
  const latest = useRef(0);
  const [error, setError] = useState<string | null>(null);
  const update = useCallback(
    (fn: (u: ProjectUpdate) => ProjectUpdate) => {
      const current = queryClient.getQueryData<ProjectInfo>(["project"]);
      if (!current) return;
      const next = fn({
        order: current.order,
        skipped: current.skipped,
        settings: current.settings,
      });
      queryClient.setQueryData<ProjectInfo>(["project"], { ...current, ...next });
      const id = ++latest.current;
      chain.current = chain.current.then(async () => {
        try {
          const saved = await saveProject(next);
          setError(null);
          if (id === latest.current) queryClient.setQueryData(["project"], saved);
        } catch (e) {
          setError(String(e));
          if (id === latest.current) void queryClient.invalidateQueries({ queryKey: ["project"] });
        }
      });
    },
    [queryClient],
  );
  return { update, error };
}

function formatTime(time: string | null): string {
  if (!time) return "no date";
  const d = new Date(time);
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

/** A circle pressed against the edge of its frame. */
function LimitingIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" aria-hidden>
      <rect x="0.75" y="0.75" width="12.5" height="12.5" rx="1.5" strokeWidth="1.5" />
      <circle cx="9" cy="7" r="3.5" strokeWidth="1.5" />
    </svg>
  );
}

function Tile({
  photo,
  info,
  view,
  index,
  dropTarget,
  onSkip,
  onDragStart,
  onDragOver,
}: {
  photo: PhotoInfo;
  info: ImageInfo | undefined;
  view: View;
  index: number | null;
  dropTarget: boolean;
  onSkip: (on: boolean) => void;
  onDragStart?: () => void;
  onDragOver?: (e: DragEvent) => void;
}) {
  const skipped = index === null;
  const classes = ["tile", skipped && "skipped", dropTarget && "drop-before"];
  return (
    <div
      className={classes.filter(Boolean).join(" ")}
      draggable={!skipped}
      onDragStart={(e) => {
        e.dataTransfer.effectAllowed = "move";
        e.dataTransfer.setData("text/plain", photo.name);
        onDragStart?.();
      }}
      onDragOver={onDragOver}
    >
      <a className="tile-image" href={imageRoute(photo.name)} title="Correct the ellipse">
        {view === "aligned" && photo.preview ? (
          <div className="thumb-frame square">
            <img src={photo.preview} loading="lazy" alt={photo.name} draggable={false} />
          </div>
        ) : info ? (
          <ThumbImage info={info} />
        ) : (
          <div className="thumb-frame square" />
        )}
        {index !== null && <span className="tile-index">{index + 1}</span>}
      </a>
      <div className="tile-caption">
        {photo.limiting && (
          <span
            className="tile-limiting"
            title={
              skipped
                ? "Its circle is close to the border: restoring this photo would zoom the crop in"
                : "Its circle is closest to the border, so this photo sets the zoom of the crop"
            }
          >
            <LimitingIcon />
          </span>
        )}
        <span className="tile-time" title={photo.name}>
          {formatTime(photo.time)}
        </span>
        {!photo.usable && <span className="badge no_ellipse">no circle</span>}
        {photo.dropped && (
          <span className="badge skip" title="Too close to the border for the common crop">
            cropped out
          </span>
        )}
        {skipped ? (
          <button onClick={() => onSkip(false)}>Restore</button>
        ) : (
          <button className="icon" title="Skip this photo" onClick={() => onSkip(true)}>
            ✕
          </button>
        )}
      </div>
    </div>
  );
}

export function Studio() {
  const project = useProject();
  const images = useImages();
  const { update, error } = useProjectUpdate();
  const [view, setView] = useView();
  const [playing, setPlaying] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [dragged, setDragged] = useState<string | null>(null);
  const [dropIndex, setDropIndex] = useState<number | null>(null);
  const [sorting, setSorting] = useState(false);
  const name = project.data?.name;
  useEffect(() => {
    if (name) document.title = `${name} · circlapse`;
  }, [name]);

  if (project.error) return <div className="message error">{String(project.error)}</div>;
  if (!project.data) return <div className="message">Loading…</div>;
  const p = project.data;
  const infos = new Map((images.data ?? []).map((i) => [i.name, i]));
  const photos = new Map(p.photos.map((ph) => [ph.name, ph]));
  const skippedSet = new Set(p.skipped);
  const included = p.order.filter((n) => !skippedSet.has(n));
  const dropped = p.photos.filter((ph) => ph.dropped).length;
  const unusable = included.filter((n) => !photos.get(n)?.usable).length;

  const skip = (name: string, on: boolean) =>
    update((u) => ({ ...u, skipped: setSkipped(u.skipped, name, on) }));

  const drop = (to: number) => {
    if (dragged === null) return;
    // `to` indexes the included photos; map it to a position in the full order
    const target = included[to];
    const from = p.order.indexOf(dragged);
    const toFull = target === undefined ? p.order.length : p.order.indexOf(target);
    update((u) => ({ ...u, order: moveItem(u.order, from, toFull) }));
    setDragged(null);
    setDropIndex(null);
  };

  const over = (i: number) => (e: DragEvent) => {
    if (dragged === null) return;
    e.preventDefault();
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const after = e.clientX > rect.left + rect.width / 2;
    setDropIndex(after ? i + 1 : i);
  };

  return (
    <div className="studio">
      <header>
        <div className="studio-title">
          <h1>{p.name}</h1>
          <span className="meta">
            {p.frames} frames
            {p.skipped.length > 0 && ` · ${p.skipped.length} skipped`}
            {dropped > 0 && ` · ${dropped} cropped out`}
            {unusable > 0 && ` · ${unusable} without circle`}
            {p.extent !== null && ` · circle fills ${Math.round(100 / p.extent)}% of the frame`}
          </span>
        </div>
        <nav className="filters">
          <button className={view === "photos" ? "active" : ""} onClick={() => setView("photos")}>
            Photos
          </button>
          <button
            className={view === "aligned" ? "active" : ""}
            onClick={() => setView("aligned")}
          >
            Aligned
          </button>
        </nav>
        <button
          title="Order the photos by date, color, shape or meaning"
          onClick={() => setSorting(true)}
        >
          Sort…
        </button>
        <span className="spacer" />
        <button disabled={p.frames === 0} onClick={() => setPlaying(true)}>
          ▶ Preview
        </button>
        <button className="primary" disabled={p.frames === 0} onClick={() => setExporting(true)}>
          Export video…
        </button>
      </header>
      {error && <div className="message error">Saving failed: {error}</div>}
      <p className="hint">
        Drag photos to reorder, ✕ skips one, click a photo to correct its
        ellipse.
      </p>

      {included.length === 0 ? (
        <div className="message">All photos are skipped.</div>
      ) : (
        <div
          className="grid studio-grid"
          onDragOver={(e) => dragged !== null && e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            if (dropIndex !== null) drop(dropIndex);
          }}
          onDragEnd={() => {
            setDragged(null);
            setDropIndex(null);
          }}
        >
          {included.map((name, i) => {
            const photo = photos.get(name);
            if (!photo) return null;
            return (
              <Tile
                key={name}
                photo={photo}
                info={infos.get(name)}
                view={view}
                index={i}
                dropTarget={dropIndex === i && dragged !== name}
                onSkip={(on) => skip(name, on)}
                onDragStart={() => setDragged(name)}
                onDragOver={over(i)}
              />
            );
          })}
          {dragged !== null && (
            <div
              className={`tile drop-end ${dropIndex === included.length ? "drop-before" : ""}`}
              onDragOver={(e) => {
                e.preventDefault();
                setDropIndex(included.length);
              }}
            />
          )}
        </div>
      )}

      {p.skipped.length > 0 && (
        <section className="skipped">
          <h2>Skipped ({p.skipped.length})</h2>
          <div className="grid studio-grid small">
            {p.skipped.map((name) => {
              const photo = photos.get(name);
              if (!photo) return null;
              return (
                <Tile
                  key={name}
                  photo={photo}
                  info={infos.get(name)}
                  view={view}
                  index={null}
                  dropTarget={false}
                  onSkip={(on) => skip(name, on)}
                />
              );
            })}
          </div>
        </section>
      )}

      {playing && (
        <Player
          project={p}
          onSkip={(name) => skip(name, true)}
          onClose={() => setPlaying(false)}
        />
      )}
      {exporting && (
        <ExportDialog project={p} update={update} onClose={() => setExporting(false)} />
      )}
      {sorting && (
        <SortDialog photos={p.photos} update={update} onClose={() => setSorting(false)} />
      )}
    </div>
  );
}
