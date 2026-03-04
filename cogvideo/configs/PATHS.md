# Config path convention

All weight paths in the YAML configs are **relative to the DimensionX repo root**, assuming you run from `cogvideo/` (e.g. `cd cogvideo && python sample_video_lowR.py`). So `..` = repo root.

| Purpose      | Path in config           | Meaning |
|-------------|---------------------------|---------|
| Main model  | `../checkpoints`           | `DimensionX/checkpoints/` with `1/mp_rank_00_model_states.pt` and `latest` |
| T5          | `../sat_weights/t5-v1_1-xxl` | T5 text encoder directory |
| VAE         | `../sat_weights/vae/3d-vae.pt` | 3D VAE checkpoint file |

Recommended layout:

```
DimensionX/
  cogvideo/
  checkpoints/
    latest            # file containing "1"
    1/
      mp_rank_00_model_states.pt
  sat_weights/
    t5-v1_1-xxl/
    vae/
      3d-vae.pt
```

If your checkpoint dir has a different name, override via `run_batch_pipeline.py --sat_checkpoint_dir <path>`; the batch script patches the config with your path.
