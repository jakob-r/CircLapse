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
- Ellipses as "filled" objects and as "outlines"

## Thoughts

- I have an old version in `./old_version` that uses opencv and edge detection but it did not work well.
- I guess a version using neural networks would be better, but I didn't find a good architecture for ellipse detection.

## Tech Stack

- Python
- uv
- and whatever packages help us