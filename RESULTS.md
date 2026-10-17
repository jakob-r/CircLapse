# Training results

One row per training run. `make experiment NOTE="what changed"` trains into `runs/<date>_<commit>/`
(checkpoints, logs, held-out CV predictions, `meta.json` with settings), makes it the current model
and adds its row here. To go back to an earlier run: `git checkout <commit>` for the code and
`make use RUN=<run>` for the model (only runs whose folder still has checkpoints).

Scores are 5-fold cross-validation on the labeled photos, held out from training, in percent:
good (hit or close in center, rectification and scale) / hit (IoU ≥ 0.9 with any annotated ellipse) /
strict (IoU ≥ 0.9 with the ellipse marked as prominent). "rounded" applies the detector's
roundness step (`DEFAULT_ROUNDNESS`). "combined" keeps the mask ellipse when it is within the
"good" tolerances of the params ellipse and uses the params ellipse otherwise (`combine_heads`);
"combined rounded" is what `circlapse detect` outputs by default. Runs on different numbers of
photos are not directly comparable.

| run | date | commit | photos | mask good / hit / strict | mask rounded good / hit / strict | params good / hit / strict | combined rounded good / hit / strict | note |
|---|---|---|---|---|---|---|---|---|
| 2026-10-15_2046_b91d670 | 2026-10-15 | b91d670+dirty | 11 | 72.7 / 54.5 / 54.5 | 81.8 / 54.5 / 54.5 | 72.7 / 36.4 / 27.3 | 81.8 / 54.5 / 54.5 | first NN, 11 photos (checkpoint overwritten) |
| 2026-10-16_0028_87be379 | 2026-10-16 | 87be379+dirty | 37 | 75.7 / 70.3 / 62.2 | 89.2 / 70.3 / 62.2 | 78.4 / 51.4 / 40.5 | 78.4 / 62.2 / 48.6 | photometric aug, ecc cap 0.6, 37 photos (checkpoint overwritten) |
| 2026-10-16_1200_c506334 | 2026-10-16 | c506334 | 37 | 70.3 / 62.2 / 54.1 | 81.1 / 62.2 / 56.8 | 70.3 / 40.5 / 32.4 | 70.3 / 54.1 / 43.2 | round synth shapes, perspective 0.01 |
| 2026-10-16_2221_18e2707 | 2026-10-16 | db26c60 | 108 | 80.6 / 68.5 / 62.0 | 85.2 / 68.5 / 62.0 | 88.9 / 55.6 / 47.2 | 88.9 / 68.5 / 59.3 | multi-target scale-tolerant loss, synth sister rings (1-4, close, drift, shaded), outline inner rim |
