import { describe, expect, it } from "vitest";
import type { PhotoInfo } from "./api";
import { moveItem, setSkipped, sortByDate } from "./projectOps";

const photo = (name: string, time: string | null): PhotoInfo => ({
  name,
  time,
  usable: true,
  dropped: false,
  limiting: false,
  preview: null,
});

describe("projectOps", () => {
  it("moves an item in front of the target", () => {
    const l = ["a", "b", "c", "d"];
    expect(moveItem(l, 0, 2)).toEqual(["b", "a", "c", "d"]);
    expect(moveItem(l, 3, 0)).toEqual(["d", "a", "b", "c"]);
    expect(moveItem(l, 1, 4)).toEqual(["a", "c", "d", "b"]);
    expect(moveItem(l, 1, 2)).toBe(l);
  });

  it("sorts by capture time with undated photos last", () => {
    const photos = [
      photo("x", "2025-07-20T09:00:00"),
      photo("y", null),
      photo("z", "2025-07-19T09:00:00"),
      photo("a", null),
    ];
    expect(sortByDate(["x", "y", "z", "a"], photos)).toEqual(["z", "x", "a", "y"]);
  });

  it("toggles exclusion", () => {
    expect(setSkipped(["a"], "b", true)).toEqual(["a", "b"]);
    expect(setSkipped(["a", "b"], "a", false)).toEqual(["b"]);
  });
});
