# Egg MRI Segmentation

This project provides Jupyter notebook pipeline for training and validating a 3D U-Net segmentation model on a single egg MRI scan. The model learns to segment three foreground classes: egg white, yolk, and air pocket against background.

The goal is to deliberately overfit one scan as a sanity check before scaling to multiple volumes. Once reliable, the same components can be lifted into a CLI for multi-case training and inference.



### Why Overfit One Case?

- Pipeline validation – quickly verifies data handling, transforms, and loss calculations.
- Capacity check – ensures the model can drive loss near zero before scaling up.
- Hyperparameter scaffolding – provides stable defaults for learning rate, loss weighting, and augmentation.
- Class coverage – guarantees the model learns to represent all egg structures, including small air pockets.


## Dataset 
- The dataset consists of a single 3D MRI scan of an egg.
- A label map was manually annotated in 3D Slicer (approximate segmentation).

## Features

### Preprocessing

- Canonical orientation (RAS), resampling to fixed spacing
Intensity normalization (percentile scaling + z-score)

- NIfTI outputs with spatial metadata preserved

### Training

- 3D U-Net `(monai.networks.nets.UNet, 1 input channel, 4 output classes)`.
- Loss: Dice + weighted cross-entropy via `monai.losses.DiceCELos`s to handle class imbalance.
- Class-balanced patch sampling with MONAI’s dataset utilities. (ensures yolk/air coverage)
- Mixed precision training using the AdamW optimizer and MONAI AMP support.
- Data preprocessing/augmentation with MONAI transforms (normalization, affine, flips).

### Inference & Postprocessing

- Sliding-window full-volume prediction
- Largest connected component filtering per class
- Per-class Dice metrics and quick slice visualizations

### Export

- Trained weights, predictions, metrics, and config snapshot



## Outputs

- `ckpts/unet_egg_single_overfit.pt` – trained model weights
- `preds/pred.nii.gz` – raw segmentation prediction
- `preds/pred_pp.nii.gz` – post-processed segmentation
- `metrics.json` – per-class Dice scores
- `config_snapshot.json` – full training configuration


## Next Steps

- This notebook can be a launchpad for a CLI
- Reuse the preprocessing, dataset, and inference transforms as standalone modules.
- Lift training, stats, and evaluation routines into CLI commands.
- Scale to multi-case training by extending the dataset list, while keeping transforms, model, and losses unchanged.
