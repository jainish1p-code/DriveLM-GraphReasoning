# Experiment log

## 2026-11-22 — CARLA pilot audit

- Dataset: DriveLM-CARLA QA archive.
- Scenario: `Town13/AccidentTwoWays`.
- Frames: 2,730.
- QA records: perception 35,309; prediction 24,208; planning 16,199; behavior 0.
- Front-camera image references: 2,730.
- Missing front images after downloading the pilot scenario: 0.
- Graph edge references: 135,052.
- Route groups: 28.
- Split: 2,098 training frames and 632 validation frames.

## Interpretation

The CARLA release provides graph-connected perception, prediction, and planning QA, but no separate behavior QA group. The primary experiment will therefore evaluate planning/action QA with and without linked graph context.
