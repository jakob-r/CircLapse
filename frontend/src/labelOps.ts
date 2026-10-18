import type { Label, ScoredEllipse, Status } from "./api";

/** Pure label edits. Status follows the content unless explicitly set by the user. */

export function addEllipse(label: Label, e: ScoredEllipse): Label {
  const ellipses = [...label.ellipses, { ...e, score: null }];
  return {
    ...label,
    ellipses,
    prominent: label.prominent ?? ellipses.length - 1,
    status: "labeled",
    source: "human",
  };
}

export function updateEllipse(label: Label, i: number, e: ScoredEllipse): Label {
  return { ...label, ellipses: label.ellipses.map((x, j) => (j === i ? e : x)), source: "human" };
}

export function deleteEllipse(label: Label, i: number): Label {
  const ellipses = label.ellipses.filter((_, j) => j !== i);
  let prominent = label.prominent;
  if (prominent === i) prominent = ellipses.length ? 0 : null;
  else if (prominent !== null && prominent > i) prominent -= 1;
  const status: Status =
    label.status === "labeled" && ellipses.length === 0 ? "unlabeled" : label.status;
  return { ...label, ellipses, prominent, status, source: "human" };
}

export function setProminent(label: Label, i: number): Label {
  return { ...label, prominent: i, status: "labeled", source: "human" };
}

/** Toggle a terminal status ("no_ellipse" / "skip"); toggling off restores the content status.
 * "no_ellipse" also removes the ellipses: the photo has none. */
export function toggleStatus(label: Label, status: "no_ellipse" | "skip"): Label {
  if (label.status === status) {
    const has = label.ellipses.length > 0;
    return {
      ...label,
      status: has ? "labeled" : "unlabeled",
      prominent: has ? (label.prominent ?? 0) : null,
    };
  }
  if (status === "no_ellipse") return { ...label, ellipses: [], status, prominent: null };
  return { ...label, status };
}
