import { thumbUrl, type ImageInfo } from "../api";
import { svgPath } from "../geometry";

/** Thumbnail with the prominent (solid) and predicted (dashed) ellipse drawn on top. */
export function ThumbImage({ info }: { info: ImageInfo }) {
  const stroke = Math.max(info.width, info.height) / 120;
  return (
    <div className="thumb-frame" style={{ aspectRatio: `${info.width} / ${info.height}` }}>
      <img src={thumbUrl(info.name)} loading="lazy" alt={info.name} draggable={false} />
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
  );
}
