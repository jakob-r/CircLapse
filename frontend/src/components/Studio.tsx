import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState, type DragEvent } from "react";
import {
  fetchSortedOrder,
  saveProject,
  useImages,
  useProject,
  type ImageInfo,
  type PhotoInfo,
  type ProjectInfo,
  type ProjectUpdate,
  type SortBy,
} from "../api";
import { moveItem, setExcluded, sortByDate } from "../projectOps";
import { imageRoute } from "../routes";
import { ExportDialog } from "./ExportDialog";
import { Player } from "./Player";
import { ThumbImage } from "./ThumbImage";

type View = "photos" | "aligned";
const VIEW_KEY = "circlapse.studio.view";

/** The grid view survives trips to the editor (which unmount the studio). */
function useView(): [View, (v: View) => void] {
  const [view, setView] = useState<View>(() => {
    try {
      return sessionStorage.getItem(VIEW_KEY) === "aligned" ? "aligned" : "photos";
    } catch {
      return "photos";
    }
  });
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
        excluded: current.excluded,
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

function Tile({
  photo,
  info,
  view,
  index,
  dropTarget,
  onExclude,
  onDragStart,
  onDragOver,
}: {
  photo: PhotoInfo;
  info: ImageInfo | undefined;
  view: View;
  index: number | null;
  dropTarget: boolean;
  onExclude: (on: boolean) => void;
  onDragStart?: () => void;
  onDragOver?: (e: DragEvent) => void;
}) {
  const excluded = index === null;
  const classes = ["tile", excluded && "excluded", dropTarget && "drop-before"];
  return (
    <div
      className={classes.filter(Boolean).join(" ")}
      draggable={!excluded}
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
        <span className="tile-time" title={photo.name}>
          {formatTime(photo.time)}
        </span>
        {!photo.usable && <span className="badge no_ellipse">no circle</span>}
        {photo.dropped && (
          <span className="badge skip" title="Too close to the border for the common crop">
            cropped out
          </span>
        )}
        {excluded ? (
          <button onClick={() => onExclude(false)}>Restore</button>
        ) : (
          <button className="icon" title="Remove from the video" onClick={() => onExclude(true)}>
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
  const [sorting, setSorting] = useState<SortBy | null>(null);
  const [sortError, setSortError] = useState<string | null>(null);
  const [loop, setLoop] = useState(false);
  const name = project.data?.name;
  useEffect(() => {
    if (name) document.title = `${name} · circlapse`;
  }, [name]);

  if (project.error) return <div className="message error">{String(project.error)}</div>;
  if (!project.data) return <div className="message">Loading…</div>;
  const p = project.data;
  const infos = new Map((images.data ?? []).map((i) => [i.name, i]));
  const photos = new Map(p.photos.map((ph) => [ph.name, ph]));
  const excludedSet = new Set(p.excluded);
  const included = p.order.filter((n) => !excludedSet.has(n));
  const dropped = p.photos.filter((ph) => ph.dropped).length;
  const unusable = included.filter((n) => !photos.get(n)?.usable).length;

  const exclude = (name: string, on: boolean) =>
    update((u) => ({ ...u, excluded: setExcluded(u.excluded, name, on) }));

  const sortBySimilarity = async (by: SortBy) => {
    setSorting(by);
    setSortError(null);
    try {
      const order = await fetchSortedOrder(by, loop);
      update((u) => ({ ...u, order }));
    } catch (e) {
      setSortError(String(e));
    } finally {
      setSorting(null);
    }
  };

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
            {p.excluded.length > 0 && ` · ${p.excluded.length} removed`}
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
        <nav className="filters sort" aria-label="Sort photos">
          <span className="meta">Sort by</span>
          <button
            disabled={sorting !== null}
            onClick={() => update((u) => ({ ...u, order: sortByDate(u.order, p.photos) }))}
          >
            Date
          </button>
          <button
            disabled={sorting !== null}
            title="Similar colors next to each other"
            onClick={() => void sortBySimilarity("color")}
          >
            {sorting === "color" ? "Sorting…" : "Color"}
          </button>
          <button
            disabled={sorting !== null}
            title="Similar patterns inside the circle next to each other"
            onClick={() => void sortBySimilarity("geometry")}
          >
            {sorting === "geometry" ? "Sorting…" : "Shape"}
          </button>
          <label
            className="loop-toggle"
            title="Color and shape: make the last photo similar to the first, for looping videos"
          >
            <input
              type="checkbox"
              aria-label="Loop"
              checked={loop}
              onChange={(e) => setLoop(e.target.checked)}
            />
            <svg viewBox="0 0 24 12" width="20" height="10" aria-hidden="true">
              <path
                d="M6 1.5C3.5 1.5 1.5 3.5 1.5 6s2 4.5 4.5 4.5c4 0 8-9 12-9 2.5 0 4.5 2 4.5 4.5s-2 4.5-4.5 4.5c-4 0-8-9-12-9z"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
              />
            </svg>
          </label>
        </nav>
        <span className="spacer" />
        <button disabled={p.frames === 0} onClick={() => setPlaying(true)}>
          ▶ Preview
        </button>
        <button className="primary" disabled={p.frames === 0} onClick={() => setExporting(true)}>
          Export video…
        </button>
      </header>
      {error && <div className="message error">Saving failed: {error}</div>}
      {sortError && <div className="message error">Sorting failed: {sortError}</div>}
      <p className="hint">
        Drag photos to reorder, ✕ removes one from the video, click a photo to correct its
        ellipse.
      </p>

      {included.length === 0 ? (
        <div className="message">All photos are removed.</div>
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
                onExclude={(on) => exclude(name, on)}
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

      {p.excluded.length > 0 && (
        <section className="excluded">
          <h2>Removed ({p.excluded.length})</h2>
          <div className="grid studio-grid small">
            {p.excluded.map((name) => {
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
                  onExclude={(on) => exclude(name, on)}
                />
              );
            })}
          </div>
        </section>
      )}

      {playing && (
        <Player
          project={p}
          onExclude={(name) => exclude(name, true)}
          onClose={() => setPlaying(false)}
        />
      )}
      {exporting && (
        <ExportDialog project={p} update={update} onClose={() => setExporting(false)} />
      )}
    </div>
  );
}
