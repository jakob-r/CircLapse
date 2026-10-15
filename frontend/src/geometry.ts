/**
 * Ellipse geometry in image pixel coordinates (x right, y down).
 * `angle_deg` is the direction of the `a` axis, measured from x towards y — the same
 * convention as Konva's `rotation` and the Python `Ellipse` model.
 */

export interface Ellipse {
  cx: number;
  cy: number;
  a: number;
  b: number;
  angle_deg: number;
}

export type Point = { x: number; y: number };

const rad = (deg: number) => (deg * Math.PI) / 180;
const deg = (r: number) => (r * 180) / Math.PI;

/** Unit vectors of the `a` axis and the `b` axis. */
export function axes(e: Ellipse): { u: Point; v: Point } {
  const t = rad(e.angle_deg);
  return { u: { x: Math.cos(t), y: Math.sin(t) }, v: { x: -Math.sin(t), y: Math.cos(t) } };
}

/** Positions of the editing handles: center, end of the `a` axis, end of the `b` axis. */
export function handles(e: Ellipse): { center: Point; major: Point; minor: Point } {
  const { u, v } = axes(e);
  return {
    center: { x: e.cx, y: e.cy },
    major: { x: e.cx + u.x * e.a, y: e.cy + u.y * e.a },
    minor: { x: e.cx + v.x * e.b, y: e.cy + v.y * e.b },
  };
}

export function moveCenter(e: Ellipse, p: Point): Ellipse {
  return { ...e, cx: p.x, cy: p.y };
}

/** Dragging the `a` handle sets both the `a` radius and the rotation. */
export function moveMajor(e: Ellipse, p: Point): Ellipse {
  const dx = p.x - e.cx;
  const dy = p.y - e.cy;
  return { ...e, a: Math.max(1, Math.hypot(dx, dy)), angle_deg: deg(Math.atan2(dy, dx)) };
}

/** Dragging the `b` handle changes only the `b` radius (projection onto the `b` axis). */
export function moveMinor(e: Ellipse, p: Point): Ellipse {
  const { v } = axes(e);
  const proj = (p.x - e.cx) * v.x + (p.y - e.cy) * v.y;
  return { ...e, b: Math.max(1, Math.abs(proj)) };
}

/** Same normalization as the backend: a >= b, angle in [0, 180). */
export function normalize(e: Ellipse): Ellipse {
  let { a, b, angle_deg } = e;
  if (b > a) {
    [a, b] = [b, a];
    angle_deg += 90;
  }
  angle_deg = ((angle_deg % 180) + 180) % 180;
  return { ...e, a, b, angle_deg };
}

/** Whether point p lies inside the ellipse. */
export function contains(e: Ellipse, p: Point): boolean {
  const { u, v } = axes(e);
  const dx = p.x - e.cx;
  const dy = p.y - e.cy;
  const s = (dx * u.x + dy * u.y) / e.a;
  const t = (dx * v.x + dy * v.y) / e.b;
  return s * s + t * t <= 1;
}

/** SVG path for drawing an ellipse (used for thumbnails). */
export function svgPath(e: Ellipse): string {
  const { u } = axes(e);
  const x0 = e.cx - u.x * e.a;
  const y0 = e.cy - u.y * e.a;
  const x1 = e.cx + u.x * e.a;
  const y1 = e.cy + u.y * e.a;
  const arc = `A ${e.a} ${e.b} ${e.angle_deg} 1 1`;
  return `M ${x0} ${y0} ${arc} ${x1} ${y1} ${arc} ${x0} ${y0} Z`;
}

/** Scale and offset that fit a (w, h) image into a (vw, vh) viewport, centered. */
export function fitView(w: number, h: number, vw: number, vh: number, margin = 0.95) {
  const scale = Math.min(vw / w, vh / h) * margin;
  return { scale, x: (vw - w * scale) / 2, y: (vh - h * scale) / 2 };
}

/** New view after zooming by `factor` around screen point `p`. */
export function zoomAt(view: { scale: number; x: number; y: number }, p: Point, factor: number) {
  const scale = view.scale * factor;
  return {
    scale,
    x: p.x - (p.x - view.x) * factor,
    y: p.y - (p.y - view.y) * factor,
  };
}

/**
 * Keep the image inside the viewport: an axis where the scaled image is smaller than the
 * viewport is centered, otherwise the image may not leave a gap at either edge.
 */
export function clampView(
  view: { scale: number; x: number; y: number },
  w: number,
  h: number,
  vw: number,
  vh: number,
) {
  const axis = (offset: number, image: number, viewport: number) =>
    image <= viewport ? (viewport - image) / 2 : Math.min(0, Math.max(viewport - image, offset));
  return {
    scale: view.scale,
    x: axis(view.x, w * view.scale, vw),
    y: axis(view.y, h * view.scale, vh),
  };
}
