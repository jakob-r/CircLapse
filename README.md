# Circlapse

Turn a pile of photos of round things — manhole covers, rose windows, clocks, signs — into a
hyperlapse in which the circle never moves.

Circlapse finds the most prominent ellipse in every photo with a neural network, warps each photo
so that the ellipse becomes a centered circle, picks the widest crop that works for all photos (optionally zoomed in further)
and encodes the sequence as mp4, webm or gif (1080 × 1080, 10 fps by default).

## Make a circlapse

```bash
circlapse make ~/Pictures/circles
```

or straight from Apple Photos:

```bash
circlapse make ~/circles --from-apple-photos --from-date 2025-07-01 --to-date 2025-07-31
```

1. Every photo directly in the folder is searched for a circle. The ones with a circle are copied
   into `<folder>/circlapse/`, renamed `YYYYMMDD-HHMMSS_<name>` and sorted by capture date.
   Your originals are never changed; photos without a circle are not copied.
   Camera RAW files (ARW, CR2, CR3, NEF, RAF, DNG, ORF, RW2) are converted to JPEG on the way in,
   keeping capture time and GPS. A RAW file is skipped when a JPEG of the same name is in the folder.
2. The studio opens in your browser:
   - drag photos to reorder them, or sort them by date (the original order), color or shape
     (both compare the aligned, cropped frames; shape ignores rotation),
   - ✕ skips a photo (it leaves the video), the "Skipped" section brings it back,
   - click a photo to correct its ellipse,
   - "Aligned" shows the rectified, cropped frames, "Preview" plays them,
   - "Export video…" picks formats, size, speed and how many photos to keep — dropping the few
     photos whose circle sits near the border lets the crop show more of the surroundings.
3. Videos are written to `<folder>/circlapse/export/` and can be downloaded from the dialog.

Run `circlapse make` on the same folder again to reopen the project, or `circlapse studio
<project folder>` to open any project. Everything is stored in the project folder:
the photos, `labels.json` with the ellipses and `project.json` with order, skipped photos and
export settings.

## Install

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), Node.js (to build the UI) and
[ffmpeg](https://ffmpeg.org/) on the `PATH`.

```bash
git clone git@github.com:jakob-r/CircLapse.git
cd CircLapse
uv sync --extra ml          # add --extra photos for Apple Photos, --extra geo for place names
make build                  # build the browser UI
uv run --extra ml circlapse make ~/Pictures/circles
```

Or simply `make studio DATA=~/Pictures/circles`.

**Model:** detection needs a trained checkpoint at `checkpoints/finetune.pt` (or pass
`--checkpoint`). Checkpoints are not part of the repository; train one as described below.

Importing from Apple Photos only reads the library. The terminal needs Full Disk Access, and
photos that only live in iCloud are downloaded through Photos.

## Other commands

| command | what it does |
|---|---|
| `circlapse render DIR` | Non-interactive: detect, align and encode all photos of `DIR` in name order |
| `circlapse organize DIR` | Sort a photo dump into trip folders by time and GPS, rejecting photos without a circle |
| `circlapse detect DIR` | Write the detected ellipses of every photo to a predictions file |
| `circlapse annotate DIR` | Label ellipses by hand to train and evaluate the detector |

`circlapse <command> --help` lists all options; `make help` lists the development targets.

## Development

```bash
make install    # Python and frontend dependencies
make dev        # API with reload on :8000, UI with hot reload on http://localhost:5173
make test       # pytest, TypeScript type check, vitest
make lint       # ruff, mypy, eslint
```

The detector is trained in two stages: pretraining on synthetic images (`make synth-train
pretrain`) and fine-tuning with 5-fold cross-validation on hand-labeled photos in `data/real`
(`make finetune`, labels made with `make annotate`). `make experiment NOTE="what changed"`
does both in a tracked run folder and adds its scores to [RESULTS.md](RESULTS.md).

- [CONCEPT.md](CONCEPT.md) — goals, constraints and design decisions
- [RESULTS.md](RESULTS.md) — detector scores per training run
- `src/circlapse/` — Python package: `nn/` network and training, `detect/` detectors,
  `render.py` alignment and encoding, `studio/` the project workflow, `annotate/` the API
- `frontend/` — React UI shared by the studio and the annotator
- `old_version/` — the earlier OpenCV edge-detection approach, kept for reference
