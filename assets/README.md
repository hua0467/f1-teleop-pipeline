# Assets

Figures and media referenced by the README and the paper.

## Generated figures

All `*.png` figures and `dataset_stats.json` are **generated** by
[`scripts/statistics.py`](../scripts/statistics.py) — do not edit them
by hand. Regenerate with:

```bash
python scripts/statistics.py \
    --dataset ./datasets/f1_vr_v2 \
    --ik-h5 "./recordings/episode_*.ik.h5" \
    --assets ./assets
```

| File | Content |
|------|---------|
| `dataset_episode_lengths.png` | frames per episode |
| `dataset_joint_distributions.png` | joint-angle distributions (17 DOF) |
| `dataset_ee_positions.png` | end-effector workspace coverage via FK |
| `ik_success_rate.png` | FK-verified vs pipeline-reported IK success |
| `ik_position_error.png` | IK position-error histogram |
| `dataset_stats.json` | machine-readable summary of all statistics |

Preview without hardware or data:

```bash
python scripts/statistics.py --demo
```

## Demo video

`assets/demo_video.mp4` — **placeholder, to be replaced** with a short
video of a complete teleoperation session (Quest 3 view + robot view).
See `demo_video_placeholder.txt` for the recording notes.
