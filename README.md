Egg MRI Segmentation (Single-Case Overfit)

This project provides Jupyter notebook pipeline for training and validating a 3D U-Net segmentation model on a single egg MRI scan. The model learns to segment three foreground classes: egg white, yolk, and air pocket against background.

The goal is to deliberately overfit one scan as a sanity check before scaling to multiple volumes. Doing so validates the preprocessing, data pipeline, model wiring, and loss setup end-to-end. 
Once reliable, the same components can be lifted into a CLI for multi-case training and inference.