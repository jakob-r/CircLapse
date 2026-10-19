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

/** Throws the error of a failed response, with FastAPI's `detail` as the message. */
async function detail(res: Response): Promise<never> {
  const text = await res.text();
  let message = text;
  try {
    message = String((JSON.parse(text) as { detail?: unknown }).detail ?? text);
  } catch {
    // Not JSON, show it as it is
  }
  throw new Error(message);
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
  skipped: string[];
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

export type SortBy = "color" | "geometry" | "meaning";
export type ColorRegion = "inside" | "outside" | "whole";

/** The project order with similar photos next to each other; `region` picks the part of the
 * frame whose colors count, `loop` also makes the last photo similar to the first. */
export async function fetchSortedOrder(
  by: SortBy,
  { region, loop }: { region: ColorRegion; loop: boolean },
): Promise<string[]> {
  const params = new URLSearchParams({ by, region, loop: String(loop) });
  const res = await fetch(`/api/sorted?${params}`);
  if (!res.ok) return detail(res);
  return ((await res.json()) as Schemas["SortedOrder"]).order;
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

// ---------------------------------------------------------------- launcher

export type LauncherInfo = Schemas["LauncherInfo"];
export type ImportRequest = Schemas["ImportRequest"];
export type ImportStatus = Schemas["ImportStatus"];

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) return detail(res);
  return (await res.json()) as T;
}

/** The start page state, or null when the server has no start page (plain annotator). */
export function useLauncher() {
  return useQuery({
    queryKey: ["launcher"],
    queryFn: async () => {
      const res = await fetch("/api/launcher");
      if (res.status === 404) return null;
      return json<LauncherInfo>(res);
    },
  });
}

export const startImport = (request: ImportRequest) =>
  post<ImportStatus>("/api/launcher/import", request);
export const openProject = (path: string) => post<LauncherInfo>("/api/launcher/open", { path });
export const closeProject = () => post<LauncherInfo>("/api/launcher/close");

export async function dismissImport(): Promise<ImportStatus> {
  return json<ImportStatus>(await fetch("/api/launcher/import", { method: "DELETE" }));
}

export function useImportStatus() {
  return useQuery({
    queryKey: ["import"],
    queryFn: async () => json<ImportStatus>(await fetch("/api/launcher/import")),
    refetchInterval: (q) => (q.state.data?.state === "running" ? 500 : false),
  });
}

// ---------------------------------------------------------------- native window

interface NativeApi {
  pick_folder(): Promise<string | null>;
}

declare global {
  interface Window {
    pywebview?: { api: NativeApi };
  }
}

/** The pywebview bridge, which appears shortly after load; null in a browser. */
export function nativeApi(): NativeApi | null {
  return window.pywebview?.api ?? null;
}
