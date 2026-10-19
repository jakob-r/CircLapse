import type { PhotoInfo } from "./api";

/** Pure edits of the frame order and the skipped set. */

/** Move the item at `from` so that it ends up in front of the item that was at `to`
 * (`to === list.length` moves it to the end). */
export function moveItem<T>(list: T[], from: number, to: number): T[] {
  if (from === to || from + 1 === to) return list;
  const out = [...list];
  const [item] = out.splice(from, 1);
  out.splice(from < to ? to - 1 : to, 0, item as T);
  return out;
}

/** Capture order; undated photos last, in name order. */
export function sortByDate(order: string[], photos: PhotoInfo[]): string[] {
  const time = new Map(photos.map((p) => [p.name, p.time]));
  return [...order].sort((x, y) => {
    const tx = time.get(x) ?? null;
    const ty = time.get(y) ?? null;
    if (tx !== ty) {
      if (tx === null) return 1;
      if (ty === null) return -1;
      if (tx < ty) return -1;
      if (tx > ty) return 1;
    }
    return x < y ? -1 : x > y ? 1 : 0;
  });
}

export function setSkipped(skipped: string[], name: string, on: boolean): string[] {
  const rest = skipped.filter((n) => n !== name);
  return on ? [...rest, name] : rest;
}
