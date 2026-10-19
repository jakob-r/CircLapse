import { useEffect, useId, useState, type ReactNode } from "react";
import {
  fetchSortedOrder,
  type ColorRegion,
  type PhotoInfo,
  type ProjectUpdate,
  type SortBy,
} from "../api";
import { sortByDate } from "../projectOps";

type Method = "date" | SortBy;
interface Choice {
  method: Method;
  region: ColorRegion;
  loop: boolean;
}

const CHOICE_KEY = "circlapse.studio.sort";
const DEFAULT_CHOICE: Choice = {
  method: "color",
  region: "whole",
  loop: false,
};

function storedChoice(): Choice {
  try {
    const saved = JSON.parse(localStorage.getItem(CHOICE_KEY) ?? "{}") as Partial<Choice>;
    return {
      method: METHODS.some((m) => m.id === saved.method) ? saved.method! : DEFAULT_CHOICE.method,
      region: REGIONS.some((r) => r.id === saved.region) ? saved.region! : DEFAULT_CHOICE.region,
      loop: saved.loop === true,
    };
  } catch {
    return DEFAULT_CHOICE;
  }
}

function storeChoice(choice: Choice) {
  try {
    localStorage.setItem(CHOICE_KEY, JSON.stringify(choice));
  } catch {
    // Remembering the choice is only a convenience
  }
}

const METHODS: { id: Method; label: string; summary: string; info: string }[] = [
  {
    id: "date",
    label: "Date",
    summary: "By capture time, the original order",
    info: "Oldest photo first. Photos without a capture time go to the end.",
  },
  {
    id: "color",
    label: "Color",
    summary: "Similar colors next to each other",
    info:
      "Compares the average L*a*b* color of the center, middle and rim of the circle and of " +
      "its surroundings in the aligned, cropped frame. Lightness counts half, so the same " +
      "scene in sun and shade still matches.",
  },
  {
    id: "geometry",
    label: "Shape",
    summary: "Similar patterns next to each other",
    info:
      "Compares the edges of the aligned, cropped frame in polar coordinates. Rotation and " +
      "mirroring don't matter, but the number of spokes and where the rings are do.",
  },
  {
    id: "meaning",
    label: "Meaning",
    summary: "Similar content next to each other",
    info:
      "A pretrained vision network (DINOv2) describes what each aligned, cropped frame shows; " +
      "photos of similar things end up together even when colors differ. The first use " +
      "downloads the model (about 85 MB) and needs the ml extra.",
  },
];

const REGIONS: { id: ColorRegion; label: string; info: string }[] = [
  {
    id: "inside",
    label: "Inside circle",
    info: "Only the colors within the circle count.",
  },
  {
    id: "outside",
    label: "Outside",
    info: "Only the surroundings of the circle count.",
  },
  {
    id: "whole",
    label: "Whole frame",
    info: "Inside and outside of the circle count.",
  },
];

/** An ⓘ that explains its neighbour on hover or keyboard focus. */
function Info({ children }: { children: ReactNode }) {
  const id = useId();
  return (
    <span className="info" tabIndex={0} aria-describedby={id}>
      ⓘ
      <span className="info-tip" role="tooltip" id={id}>
        {children}
      </span>
    </span>
  );
}

export function SortDialog({
  photos,
  update,
  onClose,
}: {
  photos: PhotoInfo[];
  update: (fn: (u: ProjectUpdate) => ProjectUpdate) => void;
  onClose: () => void;
}) {
  const [choice, setChoice] = useState<Choice>(storedChoice);
  const [sorting, setSorting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const change = (patch: Partial<Choice>) => {
    const next = { ...choice, ...patch };
    setChoice(next);
    storeChoice(next);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && !sorting && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, sorting]);

  const run = async () => {
    setError(null);
    if (choice.method === "date") {
      update((u) => ({ ...u, order: sortByDate(u.order, photos) }));
      onClose();
      return;
    }
    setSorting(true);
    try {
      const order = await fetchSortedOrder(choice.method, {
        region: choice.region,
        loop: choice.loop,
      });
      update((u) => ({ ...u, order }));
      onClose();
    } catch (e) {
      setError(String(e));
      setSorting(false);
    }
  };

  const similarity = choice.method !== "date";
  return (
    <div className="overlay" onClick={() => !sorting && onClose()}>
      <div className="dialog sort-dialog" onClick={(e) => e.stopPropagation()}>
        <h2>Sort photos</h2>
        <p className="meta">
          Photos without a circle stay at the end. You can still drag photos afterwards.
        </p>
        <fieldset className="sort-options" disabled={sorting}>
          <legend className="sr-only">Sort by</legend>
          {METHODS.map((m) => (
            <label
              key={m.id}
              className={choice.method === m.id ? "sort-option active" : "sort-option"}
            >
              <input
                type="radio"
                name="sort-method"
                checked={choice.method === m.id}
                onChange={() => change({ method: m.id })}
              />
              <span className="sort-option-text">
                <strong>{m.label}</strong>
                <span className="meta">{m.summary}</span>
              </span>
              <Info>{m.info}</Info>
            </label>
          ))}
        </fieldset>

        {choice.method === "color" && (
          <fieldset className="sort-setting" disabled={sorting}>
            <legend>Compare colors of</legend>
            <div className="segmented">
              {REGIONS.map((r) => (
                <label key={r.id} className={choice.region === r.id ? "active" : ""} title={r.info}>
                  <input
                    type="radio"
                    name="sort-region"
                    checked={choice.region === r.id}
                    onChange={() => change({ region: r.id })}
                  />
                  {r.label}
                </label>
              ))}
            </div>
          </fieldset>
        )}

        <div className="sort-setting">
          <label className="loop-toggle" aria-disabled={!similarity}>
            <input
              type="checkbox"
              aria-label="Loop"
              disabled={!similarity || sorting}
              checked={similarity && choice.loop}
              onChange={(e) => change({ loop: e.target.checked })}
            />
            <svg viewBox="0 0 24 12" width="20" height="10" aria-hidden="true">
              <path
                d="M6 1.5C3.5 1.5 1.5 3.5 1.5 6s2 4.5 4.5 4.5c4 0 8-9 12-9 2.5 0 4.5 2 4.5 4.5s-2 4.5-4.5 4.5c-4 0-8-9-12-9z"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
              />
            </svg>
            Loop
          </label>
          <Info>
            The last photo is also similar to the first, so a video that plays in a loop has no
            jump. Not available for date.
          </Info>
        </div>

        {error && <div className="export-status error">Sorting failed: {error}</div>}
        <div className="dialog-buttons">
          <button disabled={sorting} onClick={onClose}>
            Cancel
          </button>
          <button className="primary" disabled={sorting} onClick={() => void run()}>
            {sorting ? "Sorting…" : "Sort"}
          </button>
        </div>
      </div>
    </div>
  );
}
