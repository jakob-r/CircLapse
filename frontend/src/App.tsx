import { useEffect, useState } from "react";
import { Editor } from "./components/Editor";
import { Grid } from "./components/Grid";

/** Routes: `#/` grid, `#/image/<name>` editor. */
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
  const match = hash.match(/^#\/image\/(.+)$/);
  if (match?.[1]) {
    const name = decodeURIComponent(match[1]);
    return <Editor key={name} name={name} />;
  }
  return <Grid />;
}
