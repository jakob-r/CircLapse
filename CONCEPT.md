# Circlapse Concept

This python library is

- tooling to implement the circlapse algorithm and discover the best algorithms
- the circlapse algorithm itself

## High level idea

I have a folder with a bunch of photos.
Each photo shows something that a human would perceive as a circle.
E.g.: Manhole covers, round windows, clocks, signs etc.
We have to assume that the circles are not shot fully frontal and not perfectly aligned.
We also have to assume that each picture shows multiple circles, usually we are interested in the largest circle or the most prominent circle in the picture.
E.g.: A round window consists of the circular glass, the circular frame and probably some round wall.
Probably the round frame is the most prominent circle in the picture.

Goal: We want to have a hyperlapse like animation that shows one picture after another in a way that the circles are always in the same position.

Process:
1. Find the "most prominent" ellipse in each picture.
2. Transform each picture so that the ellipse becomes a circle.
3. Across all pictures, find the biggest possible cropping window that allows us to crop the circles in the same position.
4. Output the image sequence as a video.

Acceptable:
- We can drop some pictures if we cannot find a good ellipse in them.
- Detection of an ellipse should take less then 5 seconds per picture.
- Ellipses smaller then 1/3rd of the image size don't need to be detected.
- Which ellipse is the "most prominent" is subjective, also during annotation.
  Objects often consist of several (nearly) concentric ellipses, e.g. glass, frame and wall of a round window.
  If the algorithm finds one of the other annotated ellipses instead of the one marked as most prominent, or a slightly scaled down or scaled up version of it, that is still a pretty good result.
  What matters most for the animation is that center, aspect ratio and rotation are right, because then the rectification and alignment work.

## Tooling

To implement and improve the ellipse detection algorithm, we need a tool that allows us to manually annotate the ellipses in the pictures.
The annotation tool should be able to:
- Show all known pictures as thumbnails in a grid.
- Show the ellipses in the picture that the algorithm has detected and mark the most prominent one.
- Allow the user to draw a new ellipse and correct it.

We also probably need synthetic data to train the algorithm.
We need a script that generates synthetic images with ellipses in them.
Training images should feature
- Noise
- Ellipses with different scaling, rotation, position and aspect ratio
- A realistic upper limit for the eccentricity.
  Real photos are taken roughly frontal, so the ellipses are only mildly skewed:
  in our annotated data the largest eccentricity is 0.48 (minor/major axis ratio 0.88) and the median is about 0.18.
  Synthetic ellipses (prominent ones and distractors) are capped at an eccentricity of 0.6 (axis ratio 0.8), a bit above what we observed, also after the perspective warp.
  Training on much flatter ellipses wastes capacity on cases that do not occur.
- A realistic eccentricity distribution, not only a cap.
  The prominent ellipse's eccentricity is drawn from |N(0, 0.18)| and the perspective jitter during generation is 1% of the image side.
  Training augmentation keeps its stronger perspective jitter (3% synthetic, 8% real): reducing it to 1% did not make predictions rounder and scored lower in cross-validation.
  Predictions are still too eccentric (axis ratio 0.93 vs 0.98 in the labels), because fitting an ellipse to the predicted mask is noisy; the detector pulls them towards a circle instead (`Ellipse.rounded`, see RESULTS.md).
- Ellipses as "filled" objects and as "outlines"

## Thoughts

- I have an old version on the `v1` branch that uses opencv and edge detection but it did not work well.
- I guess a version using neural networks would be better, but I didn't find a good architecture for ellipse detection.

## Output video

- 10 fps
- square aspect ratio
- 1080 x 1080 pixels
- formats: mp4, gif, webm
- we can rely on ffmpeg
- cropped images can be slightly processed for nice color, brightness and contrast

## Tech Stack

- Python
- uv
- and whatever packages help us

## User friendly workflow

For users that just want to create a circlapse video from a bunch of pictures we need a nice and friendly tool.

Suggested workflow:

1. The user uses the CLI tool either on a folder or on apple photos plus a date range
2. The script then creates a folder with all candidate pictures and with all ellipses annotated
3. Then automatically the browser opens and shows a UI
4. The UI allows to remove pictures and optionally correct ellipses
5. Photos are sorted by date but the user can rearrange them
6. The use can export the video in their desired format

Usage:

```
circlapse make ~/Pictures/circles                     # a folder
circlapse make ~/circles --from-apple-photos --from-date 2025-07-01 --to-date 2025-07-31
```

`make` detects the prominent ellipse in every photo directly in the folder and copies the photos
with a circle into `<folder>/circlapse/` (the originals stay untouched; photos without a circle are
not copied), named `YYYYMMDD-HHMMSS_<name>` and sorted by capture date.
`project.json` there keeps the frame order, the removed photos and the export settings.
Then the studio opens in the browser: drag to reorder, ✕ removes a photo, clicking one opens the
ellipse editor, "Aligned" shows the rectified crops, "Preview" plays them, and "Export video"
renders mp4 / webm / gif into `<folder>/circlapse/export/`.
Running `make` again on the same folder reopens the project; `circlapse studio <project>` opens
any project folder.
