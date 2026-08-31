# Paper Draft Notes

Working notes for turning this project into a journal paper
(target venues: Sensors / Applied Sciences / Machines). Update as
experiments complete.

## Candidate Titles

1. VR-Teleop: A Lightweight Hand-Tracking Teleoperation Pipeline for
   Dual-Arm Mobile Manipulation Data Collection
2. From Hand Poses to Joint Targets: A Pure-NumPy IK Pipeline for
   LeRobot-Format Demonstration Datasets
3. Low-Cost VR Teleoperation and Behavior Cloning for a 17-DOF
   Dual-Arm Mobile Manipulator
4. Recovering Robot Demonstrations from Consumer VR Headsets: System
   Design, IK Verification, and Imitation Learning Baselines

## Core Contributions

1. **End-to-end, low-cost demonstration pipeline** — Meta Quest 3 hand
   tracking (consumer hardware) → UDP streaming → per-frame IK →
   HDF5 → LeRobot v3.0, in a single script (`06_vr_ik_record.py`).
2. **Dependency-light dual-arm IK solver** — URDF-driven kinematic
   chains plus scipy L-BFGS-B with warm-start + zero-pose dual
   initialization; runs without ROS/pinocchio/pybullet (pure
   NumPy/SciPy), easing reproducibility.
3. **Rigorous two-tier IK success metric** — (a) pipeline solve rate
   and (b) FK-verified accuracy with explicit position/orientation
   thresholds, plus an open `statistics.py` that reproduces every
   figure in the paper from the dataset alone.
4. **A published-format dataset** — 7 episodes / 3,913 frames of
   17-DOF dual-arm demonstrations, LeRobot v3.0, growing toward 10k+
   frames.
5. **Reproducibility infrastructure** — one-command training
   (`train_bc.py` + `configs/train_bc.yaml`), statistics script,
   documented hardware setup, MIT license.

## Planned Experiments

### E1. Hand–robot mapping calibration (BLOCKING for accuracy claims)

- Physically calibrate `scale` / `offset` (and the orientation
  remapping) in `hand_pose_to_robot_target`.
- Alternatives: (i) grid of known hand poses vs robot EE positions,
  least-squares fit; (ii) one-time per-user calibration pose.
- Success criterion: FK-verified position error ≤ 1 cm and orientation
  error ≤ 5° on ≥ 95% of frames.
- Current status: provisional mapping; FK-verified rate ≈ 0%
  (see `assets/ik_success_rate.png`).

### E2. Dataset expansion

- Grow from 3,913 to 10k+ frames; more tasks beyond "pick up the
  cube" (place, push, two-handed manipulation).
- Record per-episode success labels; report inter-operator variance.

### E3. IK ablations

- Warm start vs zero pose only (convergence + per-frame solve time).
- `maxiter` sweep (10/30/100) — success rate vs 30 FPS budget.
- Torso merging strategies (average vs right-dominant vs two-stage).

### E4. Behavior cloning baselines

- ACT vs Diffusion Policy on the expanded dataset (same splits).
- Metric: rollout success rate (real robot), action MSE, trajectory
  smoothness.
- Optional VLA fine-tune (π₀.₅) as an upper-bound probe.

### E5. System-level evaluation

- Teleoperation latency (UDP → joint target, end-to-end).
- Operator learning curve / usability across N operators.
- Comparison with alternative interfaces (keyboard/joystick baseline).

## Numbers to Fill In

- [ ] Calibrated mapping values (E1)
- [ ] FK-verified IK accuracy after calibration (target > 95%)
- [ ] BC baseline success rates (E4)
- [ ] End-to-end latency (E5)
- [ ] Dataset license / Hugging Face upload link
- [ ] Demo video (`assets/demo_video.mp4`)

## Related Work (reading list seeds)

- Zhao et al., ALOHA / Mobile ALOHA (low-cost teleoperation + BC)
- Chi et al., Diffusion Policy
- Zhao et al., ACT: Action Chunking with Transformers
- Hugging Face LeRobot (Cadene et al.)
- Black et al., π₀ / π₀.₅ (Physical Intelligence) — data format
  compatibility
