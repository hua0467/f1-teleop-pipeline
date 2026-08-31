# Paper Outline

One-paragraph skeleton per section; expand during writing.

## 1. Introduction

Demonstration data is the bottleneck of modern imitation learning, and
its quality depends on the teleoperation interface, the mapping from
operator input to robot targets, and the dataset format downstream
training stacks accept. This paper presents a complete, low-cost
pipeline that turns hand poses from a consumer VR headset (Meta
Quest 3) into LeRobot-format demonstrations for a 17-DOF dual-arm
mobile manipulator, using a dependency-light numerical IK solver and
verifying every step quantitatively. We report the pipeline design,
the two-tier IK success metric, a growing dataset, and behavior
cloning baselines.

## 2. Related Work

Prior work falls into three lines: (i) low-cost teleoperation systems
(ALOHA, Mobile ALOHA) that use leader-follower hardware instead of VR
hand tracking; (ii) inverse kinematics approaches ranging from
closed-form solvers for 6-DOF arms to general numerical solvers, which
rarely address dual arms sharing a torso; and (iii) dataset formats
and training stacks for VLA models (LeRobot, π₀.₅), which define the
compatibility target of any collection system. Our work combines a
consumer VR interface with a URDF-driven, dependency-light dual-arm
solver and publishes the data in a training-ready format.

## 3. System Design

The system comprises four modules: a UDP hand-tracking receiver, a
URDF-driven kinematics engine, an HDF5/LeRobot recording and
conversion layer, and a statistics/verification tool. The kinematic
engine parses the robot URDF into per-arm chains, evaluates forward
kinematics by chained 4×4 transforms, and solves IK with L-BFGS-B
under joint limits using warm-start plus zero-pose dual
initialization; shared torso joints are merged by averaging. The
converter emits LeRobot v3.0 Parquet episodes with the mandatory
`features` metadata, and the statistics module recomputes every
reported metric from the dataset alone.

## 4. Data Collection and Dataset

Demonstrations are collected by an operator wearing the Quest 3, whose
tracked hands are streamed at 30 FPS; each frame yields a 17-DOF state
and action pair plus raw hand landmarks. The current dataset contains
7 episodes and 3,913 frames of a pick-and-place task and will be
expanded beyond 10k frames with success labels and additional tasks.
We describe the HDF5 and LeRobot layouts, the hand-to-robot coordinate
mapping (and its calibration), and the integrity-validation procedure.

## 5. Experiments

Experiments address four questions: (i) calibration of the hand–robot
mapping and its effect on FK-verified IK accuracy; (ii) IK solver
ablation — warm start vs zero pose, optimizer budget vs success rate;
(iii) dataset-scale effects on behavior cloning; and (iv) end-to-end
system metrics including teleoperation latency and operator usability.
Each experiment lists its hardware, hyperparameters, and evaluation
protocol so all results can be reproduced with the scripts in this
repository.

## 6. Discussion

We discuss the two-tier IK metric — why pipeline solve rate alone is
insufficient and how FK verification exposes mapping errors that would
otherwise stay hidden — the trade-offs of a pure-NumPy solver versus
ROS-integrated alternatives, the sources of demonstration noise
(tracking jitter, uncalibrated offsets, torso averaging), and the
limitations of a single-task, single-operator dataset. Failure cases
and the conditions under which the pipeline degrades are stated
explicitly.

## 7. Conclusion

This paper contributes an end-to-end, reproducible VR teleoperation
pipeline for dual-arm mobile manipulators that runs on consumer
hardware and standard scientific Python, together with a rigorous IK
verification methodology and a LeRobot-format dataset. We release the
code and data openly and outline future work on calibrated mappings,
multi-task expansion, and VLA fine-tuning.
