from pathlib import Path

root = Path("/mnt/vast-nhr/projects/nim00018/tom/scandiff")
for scale in ("1k", "10k", "50k", "100k", "200k"):
    name = f"oursynth_v2_{scale}"
    for group, split, subjects in (
        ("train_datasets", "train", 5),
        ("val_datasets", "valid", 16),
        ("test_datasets", "test", 16),
    ):
        text = f"""{name}:
  _target_: src.data.components.mit1003.mit1003.MIT1003Dataset
  name: {name}
  root_path: ./data/thesis_v2/{name}
  task: What do you see in the image?
  split: {split}
  num_subjects: {subjects}
  time_in_ms: false
  use_abs_coords: true
  task_embeddings_file: task_embeddings.npy
  img_features_dir: dinov2_base_timm_image_features
"""
        path = root / "configs/data" / group / f"{name}.yaml"
        path.write_text(text)
        print(path)
