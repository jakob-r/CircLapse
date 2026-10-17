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

// ---------------------------------------------------------------- studio

export type PhotoInfo = Schemas["PhotoInfo"];
export type Format = NonNullable<Schemas["ExportSettings"]["formats"]>[number];
export type ExportSettings = Required<Schemas["ExportSettings"]>;
export type ProjectInfo = Omit<Schemas["ProjectInfo"], "settings"> & { settings: ExportSettings };
export type ExportStatus = Schemas["ExportStatus"];
export const FORMATS: Format[] = ["mp4", "webm", "gif"];

export interface ProjectUpdate {
  order: string[];
  excluded: string[];
  settings: ExportSettings;
}

function toProject(p: Schemas["ProjectInfo"]): ProjectInfo {
  return { ...p, settings: { ...p.settings, formats: p.settings.formats ?? [] } };
}

/** The studio project, or null when the server is the plain annotator. */
export function useProject() {
  return useQuery({
    queryKey: ["project"],
    queryFn: async () => {
      const res = await fetch("/api/project");
      if (res.status === 404) return null;
      return toProject(await json<Schemas["ProjectInfo"]>(res));
    },
  });
}

export async function saveProject(update: ProjectUpdate): Promise<ProjectInfo> {
  const res = await fetch("/api/project", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(update),
  });
  return toProject(await json<Schemas["ProjectInfo"]>(res));
}

export type SortBy = "color" | "geometry";

/** The project order with similar photos next to each other; `loop` also makes the last
 * photo similar to the first. */
export async function fetchSortedOrder(by: SortBy, loop: boolean): Promise<string[]> {
  const res = await fetch(`/api/sorted?by=${by}&loop=${loop}`);
  return (await json<Schemas["SortedOrder"]>(res)).order;
}

export async function startExport(settings: ExportSettings): Promise<ExportStatus> {
  const res = await fetch("/api/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings),
  });
  return json<ExportStatus>(res);
}

export function useExportStatus() {
  return useQuery({
    queryKey: ["export"],
    queryFn: async () => json<ExportStatus>(await fetch("/api/export")),
    refetchInterval: (q) => (q.state.data?.state === "running" ? 1000 : false),
  });
}

export const exportFileUrl = (name: string) => url("export/file", name);
