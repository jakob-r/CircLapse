import { describe, expect, it } from "vitest";
import type { Label } from "./api";
import { addEllipse, deleteEllipse, setProminent, toggleStatus } from "./labelOps";

const e = { cx: 1, cy: 2, a: 3, b: 2, angle_deg: 0 };
const empty: Label = { ellipses: [], prominent: null, status: "unlabeled", source: "human" };

describe("labelOps", () => {
  it("first ellipse becomes prominent and labels the image", () => {
    const l = addEllipse(empty, e);
    expect(l.prominent).toBe(0);
    expect(l.status).toBe("labeled");
    expect(addEllipse(l, e).prominent).toBe(0);
  });

  it("keeps the prominent index pointing at the same ellipse on delete", () => {
    let l = addEllipse(addEllipse(addEllipse(empty, e), e), e);
    l = setProminent(l, 2);
    expect(deleteEllipse(l, 0).prominent).toBe(1);
    expect(deleteEllipse(l, 2).prominent).toBe(0);
    expect(deleteEllipse(deleteEllipse(deleteEllipse(l, 0), 0), 0)).toMatchObject({
      prominent: null,
      status: "unlabeled",
    });
  });

  it("no_ellipse removes the ellipses and toggles off", () => {
    const l = addEllipse(empty, e);
    const n = toggleStatus(l, "no_ellipse");
    expect(n).toMatchObject({ status: "no_ellipse", prominent: null, ellipses: [] });
    expect(toggleStatus(n, "no_ellipse")).toMatchObject({ status: "unlabeled", prominent: null });
    expect(toggleStatus(toggleStatus(empty, "skip"), "skip").status).toBe("unlabeled");
  });
});
