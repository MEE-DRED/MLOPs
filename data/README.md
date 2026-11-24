# Data

The dataset is **not stored in Git.** Obtain `retinamnist_224.npz` (RetinaMNIST, size 224, MedMNIST v2):

1. **Colab (recommended, nothing is downloaded to your laptop):** run the notebook. `src.data_acquisition.ensure_dataset()`
   looks for the file in Drive, then downloads it with the official `medmnist` package.
2. **Manual:** download from https://medmnist.com/ and place it at `data/retinamnist_224.npz` (or in
   `MyDrive/DR_MLOps/data/` for Colab). Override the location with `DR_DATASET_NPZ`.

Expected contents (verified on the supplied file): `train_images` (1080,224,224,3) uint8, `val_images` (120,...), `test_images` (400,...)
and matching `*_labels` (N,1) with values 0-4.

`retraining_uploads/` holds user-uploaded labelled images (created at runtime, ignored by Git).

## `train/`, `validation/`, `test/`
RetinaMNIST ships as ONE `.npz` archive; the official train/validation/test split is applied in memory by `src/data_acquisition.py`, so these folders intentionally contain
no images (only `.gitkeep`). They exist to match the directory layout required by the assignment. The raw dataset must not be committed.
