import { useQuery } from "@tanstack/react-query";
import type { components } from "./api-types";
import type { Ellipse, Point } from "./geometry";

type Schemas = components["schemas"];
export type ImageInfo = Schemas["ImageInfo"];
export type Status = ImageInfo["status"];
export type ScoredEllipse = Ellipse & { score?: number | null };

/** ImageLabel with all defaults filled in. */
export interface Label {
  ellipses: ScoredEllipse[];
  prominent: number | null;
  status: Status;
  source: NonNullable<Schemas["ImageLabel"]["source"]>;
}

const url = (path: string, name: string) => `/api/${path}/${encodeURIComponent(name)}`;
export const thumbUrl = (name: string) => url("thumb", name);
export const imageUrl = (name: string) => url("image", name);

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return (await res.json()) as T;
}

function toLabel(l: Schemas["ImageLabel"] | null): Label | null {
  if (!l) return null;
  return {
    ellipses: l.ellipses ?? [],
    prominent: l.prominent ?? null,
    status: l.status ?? "unlabeled",
    source: l.source ?? "human",
  };
}

export const fetchLabel = async (name: string) =>
  toLabel(await json<Schemas["ImageLabel"]>(await fetch(url("labels", name)))) as Label;

export const fetchPrediction = async (name: string) =>
  toLabel(await json<Schemas["ImageLabel"] | null>(await fetch(url("predictions", name))));

export async function saveLabel(name: string, label: Label): Promise<void> {
  await json(
    await fetch(url("labels", name), {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(label),
    }),
  );
}

export async function fitEllipse(points: Point[]): Promise<Ellipse> {
  const res = await fetch("/api/fit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ points: points.map((p) => [p.x, p.y]) }),
  });
  return json<Ellipse>(res);
}

export function useImages() {
  return useQuery({
    queryKey: ["images"],
    queryFn: async () => json<ImageInfo[]>(await fetch("/api/images")),
  });
}

export function usePrediction(name: string) {
  return useQuery({ queryKey: ["prediction", name], queryFn: () => fetchPrediction(name) });
}
