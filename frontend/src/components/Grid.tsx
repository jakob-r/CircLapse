import { useState } from "react";
import { thumbUrl, useImages, type ImageInfo, type Status } from "../api";
import { imageRoute } from "../routes";
import { svgPath } from "../geometry";

const FILTERS: (Status | "all")[] = ["all", "unlabeled", "labeled", "no_ellipse", "skip"];

function Thumb({ info }: { info: ImageInfo }) {
  const stroke = Math.max(info.width, info.height) / 120;
  return (
    <a className="thumb" href={imageRoute(info.name)} title={info.name}>
      <div className="thumb-frame" style={{ aspectRatio: `${info.width} / ${info.height}` }}>
        <img src={thumbUrl(info.name)} loading="lazy" alt={info.name} />
        <svg viewBox={`0 0 ${info.width} ${info.height}`} preserveAspectRatio="none">
          {info.predicted && (
            <path
              d={svgPath(info.predicted)}
              className="pred"
              strokeWidth={stroke}
              strokeDasharray={`${stroke * 3} ${stroke * 2}`}
            />
          )}
          {info.prominent && (
            <path d={svgPath(info.prominent)} className="prominent" strokeWidth={stroke} />
          )}
        </svg>
      </div>
      <div className="thumb-caption">
        <span className={`badge ${info.status}`}>{info.status.replace("_", " ")}</span>
        <span className="name">{info.name}</span>
      </div>
    </a>
  );
}

export function Grid() {
  const { data, error, isLoading } = useImages();
  const [filter, setFilter] = useState<Status | "all">("all");
  if (isLoading) return <div className="message">Loading…</div>;
  if (error) return <div className="message error">{String(error)}</div>;
  const images = data ?? [];
  const counts = Object.fromEntries(
    FILTERS.map((f) => [f, f === "all" ? images.length : images.filter((i) => i.status === f).length]),
  );
  const shown = filter === "all" ? images : images.filter((i) => i.status === filter);

  return (
    <div className="grid-page">
      <header>
        <h1>Circlapse annotator</h1>
        <nav className="filters">
          {FILTERS.map((f) => (
            <button key={f} className={f === filter ? "active" : ""} onClick={() => setFilter(f)}>
              {f.replace("_", " ")} <span className="count">{counts[f]}</span>
            </button>
          ))}
        </nav>
      </header>
      {shown.length === 0 ? (
        <div className="message">No images.</div>
      ) : (
        <div className="grid">
          {shown.map((info) => (
            <Thumb key={info.name} info={info} />
          ))}
        </div>
      )}
    </div>
  );
}
