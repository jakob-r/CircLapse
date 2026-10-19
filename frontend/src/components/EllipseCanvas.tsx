import type Konva from "konva";
import { useEffect, useRef, useState } from "react";
import { Circle, Ellipse as KEllipse, Image as KImage, Layer, Line, Shape, Stage } from "react-konva";
import type { ScoredEllipse } from "../api";
import {
  clampView,
  fitView,
  handles,
  moveCenter,
  moveMajor,
  moveMinor,
  zoomAt,
  type Ellipse,
  type Point,
} from "../geometry";

export type View = { scale: number; x: number; y: number };

interface Props {
  image: HTMLImageElement | undefined;
  /** Full-resolution image size; all coordinates are in this space. */
  width: number;
  height: number;
  ellipses: ScoredEllipse[];
  prominent: number | null;
  predictions: ScoredEllipse[];
  selected: number | null;
  drawing: boolean;
  drawPoints: Point[];
  drawPreview: Ellipse | null;
  fitSignal: number;
  /** Corners of the video frame; the photo outside it is dimmed. */
  crop?: Point[];
  onSelect: (i: number | null) => void;
  onChange: (i: number, e: Ellipse) => void;
  onAcceptPrediction: (e: Ellipse) => void;
  onAddPoint: (p: Point) => void;
}

const MAX_SCALE = 40;

const COLORS = {
  prominent: "#22e07a",
  other: "#ffb020",
  prediction: "#4cc3ff",
  draw: "#ff4fd8",
};

function useSize(ref: React.RefObject<HTMLDivElement | null>) {
  const [size, setSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    if (!ref.current) return;
    const obs = new ResizeObserver(([entry]) => {
      if (entry) setSize({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    obs.observe(ref.current);
    return () => obs.disconnect();
  }, [ref]);
  return size;
}

export function EllipseCanvas(props: Props) {
  const { width, height, ellipses, prominent, selected, drawing } = props;
  const containerRef = useRef<HTMLDivElement>(null);
  const size = useSize(containerRef);
  // The view is fitted to the viewport until the user zooms or pans; resizing the viewport
  // or a new fitSignal resets it to the fitted view.
  const fitKey = `${width}x${height}@${size.width}x${size.height}#${props.fitSignal}`;
  const [userView, setUserView] = useState<{ key: string; view: View }>();
  const view =
    userView?.key === fitKey
      ? userView.view
      : fitView(width, height, size.width || 1, size.height || 1);
  const clamp = (v: View) => clampView(v, width, height, size.width, size.height);
  const setView = (fn: (v: View) => View) =>
    setUserView({ key: fitKey, view: clamp(fn(view)) });
  const minScale = fitView(width, height, size.width || 1, size.height || 1).scale * 0.5;

  const px = 1 / view.scale; // one screen pixel in image coordinates

  const onWheel = (evt: Konva.KonvaEventObject<WheelEvent>) => {
    evt.evt.preventDefault();
    const pointer = evt.target.getStage()?.getPointerPosition();
    if (!pointer) return;
    const { deltaX, deltaY, ctrlKey, metaKey } = evt.evt;
    if (ctrlKey || metaKey) {
      // Trackpad pinch (reported as ctrl+wheel) or cmd/ctrl + mouse wheel: zoom at the pointer.
      setView((v) => {
        const factor = Math.min(
          Math.max(Math.exp(-deltaY * 0.01), minScale / v.scale),
          MAX_SCALE / v.scale,
        );
        return zoomAt(v, pointer, factor);
      });
    } else {
      // Two-finger trackpad scroll: pan.
      setView((v) => ({ ...v, x: v.x - deltaX, y: v.y - deltaY }));
    }
  };

  const onStageClick = (evt: Konva.KonvaEventObject<MouseEvent>) => {
    if (evt.evt.button !== 0) return;
    const stage = evt.target.getStage();
    if (!stage) return;
    if (drawing) {
      const p = stage.getRelativePointerPosition();
      if (p) props.onAddPoint(p);
    } else if (evt.target === stage || evt.target.name() === "image") {
      props.onSelect(null);
    }
  };

  const handle = (key: string, p: Point, color: string, onMove: (p: Point) => void) => (
    <Circle
      key={key}
      x={p.x}
      y={p.y}
      radius={6 * px}
      fill={color}
      stroke="#000"
      strokeWidth={1.5 * px}
      draggable
      onDragMove={(e) => onMove({ x: e.target.x(), y: e.target.y() })}
      onMouseEnter={(e) => {
        const c = e.target.getStage()?.container();
        if (c) c.style.cursor = "grab";
      }}
      onMouseLeave={(e) => {
        const c = e.target.getStage()?.container();
        if (c) c.style.cursor = "";
      }}
    />
  );

  const sel = selected !== null ? ellipses[selected] : undefined;

  return (
    <div ref={containerRef} className={`canvas ${drawing ? "drawing" : ""}`}>
      <Stage
        width={size.width}
        height={size.height}
        scaleX={view.scale}
        scaleY={view.scale}
        x={view.x}
        y={view.y}
        draggable
        dragBoundFunc={(pos) => {
          const v = clamp({ ...view, x: pos.x, y: pos.y });
          return { x: v.x, y: v.y };
        }}
        onDragEnd={(e) => {
          if (e.target === e.target.getStage()) {
            setView((v) => ({ ...v, x: e.target.x(), y: e.target.y() }));
          }
        }}
        onWheel={onWheel}
        onClick={onStageClick}
      >
        <Layer>
          {props.image && (
            <KImage name="image" image={props.image} width={width} height={height} />
          )}
          {props.crop && (
            <>
              <Shape
                listening={false}
                fill="rgba(0, 0, 0, 0.55)"
                sceneFunc={(ctx, shape) => {
                  const c = props.crop!;
                  const g = ctx._context;
                  g.save();
                  g.beginPath();
                  g.rect(0, 0, width, height);
                  g.clip();
                  // even-odd within the photo: the frame is a hole in the dimmed rectangle
                  g.beginPath();
                  g.rect(0, 0, width, height);
                  g.moveTo(c[0]!.x, c[0]!.y);
                  for (const p of c.slice(1)) g.lineTo(p.x, p.y);
                  g.closePath();
                  g.fillStyle = shape.fill() as string;
                  g.fill("evenodd");
                  g.restore();
                }}
              />
              <Line
                listening={false}
                points={props.crop.flatMap((p) => [p.x, p.y])}
                closed
                stroke="rgba(255, 255, 255, 0.8)"
                strokeWidth={px}
                dash={[6 * px, 4 * px]}
              />
            </>
          )}
        </Layer>
        <Layer>
          {props.predictions.map((e, i) => (
            <KEllipse
              key={`pred-${i}`}
              x={e.cx}
              y={e.cy}
              radiusX={e.a}
              radiusY={e.b}
              rotation={e.angle_deg}
              stroke={COLORS.prediction}
              strokeWidth={2 * px}
              hitStrokeWidth={12 * px}
              dash={[10 * px, 6 * px]}
              listening={!drawing}
              onClick={(evt) => {
                evt.cancelBubble = true;
                props.onAcceptPrediction(e);
              }}
            />
          ))}
          {ellipses.map((e, i) => (
            <KEllipse
              key={i}
              x={e.cx}
              y={e.cy}
              radiusX={e.a}
              radiusY={e.b}
              rotation={e.angle_deg}
              stroke={i === prominent ? COLORS.prominent : COLORS.other}
              strokeWidth={(i === selected ? 3.5 : 2) * px}
              hitStrokeWidth={12 * px}
              shadowColor="#000"
              shadowBlur={3 * px}
              listening={!drawing}
              draggable={i === selected}
              onClick={(evt) => {
                evt.cancelBubble = true;
                props.onSelect(i);
              }}
              onDragMove={(evt) =>
                props.onChange(i, moveCenter(e, { x: evt.target.x(), y: evt.target.y() }))
              }
            />
          ))}
          {sel &&
            !drawing &&
            (() => {
              const h = handles(sel);
              const i = selected!;
              return [
                handle("center", h.center, "#fff", (p) => props.onChange(i, moveCenter(sel, p))),
                handle("major", h.major, COLORS.prominent, (p) =>
                  props.onChange(i, moveMajor(sel, p)),
                ),
                handle("minor", h.minor, COLORS.other, (p) => props.onChange(i, moveMinor(sel, p))),
              ];
            })()}
          {drawing && props.drawPreview && (
            <KEllipse
              x={props.drawPreview.cx}
              y={props.drawPreview.cy}
              radiusX={props.drawPreview.a}
              radiusY={props.drawPreview.b}
              rotation={props.drawPreview.angle_deg}
              stroke={COLORS.draw}
              strokeWidth={2 * px}
              dash={[6 * px, 4 * px]}
              listening={false}
            />
          )}
          {drawing &&
            props.drawPoints.map((p, i) => (
              <Circle
                key={`pt-${i}`}
                x={p.x}
                y={p.y}
                radius={4 * px}
                fill={COLORS.draw}
                stroke="#000"
                strokeWidth={px}
                listening={false}
              />
            ))}
        </Layer>
      </Stage>
    </div>
  );
}
