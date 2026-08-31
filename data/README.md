# Data

This directory is reserved for linked or copied datasets.

By default, raw recordings and LeRobot datasets live outside Git and
are produced by the pipeline scripts:

| Location | Content | Git |
|----------|---------|-----|
| `recordings/` | raw HDF5 episodes (`episode_XXXXXX.h5`, `*.ik.h5`) | ignored |
| `datasets/<name>/` | LeRobot v3.0 datasets | ignored |

## Releasing a dataset

1. Generate it with the recording/conversion scripts (see README).
2. Optionally upload to the Hugging Face Hub:
   ```bash
   huggingface-cli upload <your-org>/<dataset-name> ./datasets/<name> --repo-type dataset
   ```
3. Link the release here (e.g. a `dataset_links.md` file) and update
   the README's dataset table.

> ⚠️ Do not commit recorded demonstration data to this repository.
