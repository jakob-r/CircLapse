import { describe, expect, it } from "vitest";
import {
  contains,
  fitView,
  handles,
  moveMajor,
  moveMinor,
  normalize,
  zoomAt,
  type Ellipse,
} from "./geometry";

const e: Ellipse = { cx: 100, cy: 50, a: 40, b: 20, angle_deg: 90 };

describe("handles", () => {
  it("places the major handle along the rotated a axis (y down)", () => {
    const h = handles(e);
    expect(h.major.x).toBeCloseTo(100);
    expect(h.major.y).toBeCloseTo(90);
    expect(h.minor.x).toBeCloseTo(80);
    expect(h.minor.y).toBeCloseTo(50);
  });

  it("round-trips: dragging a handle to its own position changes nothing", () => {
    const h = handles(e);
    const m = moveMinor(moveMajor(e, h.major), h.minor);
    expect(m.a).toBeCloseTo(e.a);
    expect(m.b).toBeCloseTo(e.b);
    expect(m.angle_deg).toBeCloseTo(e.angle_deg);
  });
});

describe("moveMajor / moveMinor", () => {
  it("rotates and resizes with the major handle", () => {
    const m = moveMajor(e, { x: 130, y: 80 });
    expect(m.a).toBeCloseTo(Math.hypot(30, 30));
    expect(m.angle_deg).toBeCloseTo(45);
    expect(m.b).toBe(e.b);
  });

  it("only uses the component along the b axis for the minor handle", () => {
    const m = moveMinor(e, { x: 70, y: 1000 });
    expect(m.b).toBeCloseTo(30);
    expect(m.a).toBe(e.a);
  });
});

describe("normalize", () => {
  it("swaps axes so that a >= b and wraps the angle", () => {
    const n = normalize({ cx: 0, cy: 0, a: 10, b: 20, angle_deg: 170 });
    expect(n.a).toBe(20);
    expect(n.b).toBe(10);
    expect(n.angle_deg).toBeCloseTo(80);
    expect(normalize({ ...e, angle_deg: -30 }).angle_deg).toBeCloseTo(150);
  });
});

describe("contains", () => {
  it("respects rotation", () => {
    expect(contains(e, { x: 100, y: 85 })).toBe(true);
    expect(contains(e, { x: 135, y: 50 })).toBe(false);
  });
});

describe("view", () => {
  it("fits and zooms around a fixed point", () => {
    const v = fitView(200, 100, 400, 400, 1);
    expect(v.scale).toBe(2);
    expect(v.y).toBe(100);
    const z = zoomAt(v, { x: 50, y: 60 }, 2);
    // the image point under the cursor stays under the cursor
    const before = { x: (50 - v.x) / v.scale, y: (60 - v.y) / v.scale };
    const after = { x: (50 - z.x) / z.scale, y: (60 - z.y) / z.scale };
    expect(after.x).toBeCloseTo(before.x);
    expect(after.y).toBeCloseTo(before.y);
  });
});
