# DriveLM Graph Reasoning

## Research question

Does adding graph-connected perception and prediction context improve a vision-language model's planning/action answers compared with using the front-camera image alone?

## Experimental conditions

1. Image-only planning: front image + planning question.
2. Gold graph context: front image + linked ground-truth upstream QA answers + planning question.
3. Predicted graph context: front image + model-generated upstream answers + planning question.

The primary model output remains free-form text. Action intent (for example, stop, brake, maintain speed, or accelerate) is an evaluation analysis, not a required canonical output format.

## Data

The project uses DriveLM-CARLA QA annotations and the corresponding PDM-Lite CARLA RGB frames. Large datasets, model caches, and checkpoints are intentionally excluded from Git.

## Current status

- CARLA graph QA archive verified.
- Pilot scenario: Town13 / AccidentTwoWays.
- Pilot size: 2,730 frames.
- Route-level split: 2,098 training frames and 632 validation frames.
- Front-camera image coverage: 100% for the pilot.
- Planning QA is the target task; the released CARLA files contain no separate behavior QA group.
