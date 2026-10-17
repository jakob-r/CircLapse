import { useEffect, useState } from "react";
import { useProject } from "./api";
import { Editor } from "./components/Editor";
import { Grid } from "./components/Grid";
import { Studio } from "./components/Studio";

/** Routes: `#/` studio (for a project) or annotator grid, `#/image/<name>` editor. */
function useHash(): string {
  const [hash, setHash] = useState(window.location.hash);
  useEffect(() => {
    const onChange = () => setHash(window.location.hash);
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return hash;
}

export function App() {
  const hash = useHash();
  const project = useProject();
  const match = hash.match(/^#\/image\/(.+)$/);
  if (match?.[1]) {
    const name = decodeURIComponent(match[1]);
    return <Editor key={name} name={name} />;
  }
  if (project.isLoading) return <div className="message">Loading…</div>;
  return project.data === null ? <Grid /> : <Studio />;
}
